!pip install torch > /dev/null

!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py > /dev/null

!python pytorch-xla-env-setup.py --version 20200515 --apt-packages libomp5 libopenblas-dev > /dev/null

!pip install transformers > /dev/null

!pip install pandarallel > /dev/null

!pip install emoji

!pip install num2words

!pip install --upgrade language_tool_python

!pip install --upgrade turkishnlp

!pip install --upgrade git+https://github.com/pytorch/contrib.git > /dev/null
import os

os.environ['XLA_USE_BF16'] = "1"



from pathlib import Path

import pandas as pd

import gc

import torch

import json

from pandarallel import pandarallel

pandarallel.initialize(nb_workers=8, progress_bar=False)



from transformers import AutoModel, AutoTokenizer, AutoConfig

import torch_xla.core.xla_model as xm

import torch_xla.distributed.xla_multiprocessing as xmp



import data_cleaning as clean

import models

import config

import utility
MODEL = 'xlm-roberta-large'



root_dir = Path("../input")



# *Data access

test_file_dir = Path(root_dir, "jigsaw-multilingual-toxic-comment-classification")



# *Model paths

model_dir = Path(root_dir, MODEL)

checkpoint_dir = (model_dir/'best_model.bin')

config_path = (model_dir/'config.json')

out_dir = Path('../output')



# *Files

test_file = "test.csv"

# yandex translated test file

test_eng_file1 = "jigsaw-multilingual-toxic-test-translated/jigsaw_miltilingual_test_translated.csv"

# google translated test file

test_eng_file2 = "test-en-df/test_en.csv"



sub_file = "jigsaw-toxic/sample_submission.csv"

open_subtitles_file = 'open-subtitles-toxic/open-subtitles-synthesic.csv'



LANGS = {

    'en': 'english',

    'it': 'italian', 

    'fr': 'french', 

    'es': 'spanish',

    'tr': 'turkish', 

    'ru': 'russian',

    'pt': 'portuguese'

}



# read config file

with open(config_path) as f:

    model_config = json.load(f)



MODEL_VERSION = model_config['model_version']

MODEL_PREFIX = model_config['model_prefix']

MAX_LENGTH = model_config['max_len']



english_pipeline = model_config['english pipeline']

output_hidden_states = model_config['output_hidden_states']
# Read the data and external sources

dir_paths = {'base_dir': test_file_dir, 'base_t_dir': root_dir, 'test_file':test_file, 

             'test_file_translated1': test_eng_file1, 'test_file_translated2': test_eng_file2}

_, _, test, _ = utility.read_data(dir_paths, list(LANGS.keys()), english_pipeline=english_pipeline)



if english_pipeline:

    test_yandex, test_google = test[0], test[1]

    test = test_yandex.append(test_google) 

#     df_test_yandex = clean.clean_data(test_yandex, input_cols_test)

#     df_test_google = clean.clean_data(test_google, input_cols_test)



input_cols_test = ['content']

df_test = clean.clean_data(test, input_cols_test, True)

test.head()
tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
test_dataset = models.DatasetRetriever(

    tokenizer,

    labels_or_ids=test.index.values, 

    comment_texts=test[input_cols_test].values, 

    langs=test['lang'].values,

    maxlen=MAX_LENGTH,

    use_train_transforms=False,

    test=True

)



del test

gc.collect();
transformer = AutoModel.from_config(AutoConfig.from_pretrained(str(model_dir)))

net = models.ToxicSimpleNNModel(transformer, config.TrainGlobalConfig)

checkpoint = torch.load(checkpoint_dir, map_location=torch.device('cpu'))

net.load_state_dict(checkpoint);

config.TrainGlobalConfig.train_lenght = len(test_dataset)



checkpoint = None

del checkpoint
import warnings

warnings.filterwarnings("ignore")



def _mp_fn(rank, flags):

    device = xm.xla_device()

    net.to(device)

    

    test_sampler = torch.utils.data.distributed.DistributedSampler(

        test_dataset,

        num_replicas=xm.xrt_world_size(),

        rank=xm.get_ordinal(),

        shuffle=False

    )

    test_loader = torch.utils.data.DataLoader(

        test_dataset,

        batch_size=16,

        sampler=test_sampler,

        pin_memory=False,

        drop_last=False,

        num_workers=1

    )

    

    fitter = models.TPUFitter(model=net, device=device, config=config.TrainGlobalConfig, 

                              base_model_path=model_dir, model_name=MODEL, model_prefix=MODEL_PREFIX,

                              model_version=MODEL_VERSION, 

                              out_path=out_dir)

    

    fitter.run_inference(test_loader)
%%time



FLAGS={}

xmp.spawn(_mp_fn, args=(FLAGS,), nprocs=8, start_method='fork')
submission = pd.concat([pd.read_csv(path) for path in (out_dir/'node_submissions').glob('*.csv')]).groupby('id').mean()

submission['toxic'].hist(bins=100)
submission.iloc[32]
submission
submission.to_csv('submission.csv')