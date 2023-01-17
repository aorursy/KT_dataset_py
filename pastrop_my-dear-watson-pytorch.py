# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
os.environ["WANDB_API_KEY"] = "0" ## to silence warning
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except ValueError:
    strategy = tf.distribute.get_strategy() # for CPU and single GPU
    print('Number of replicas:', strategy.num_replicas_in_sync)
train = pd.read_csv("../input/contradictory-my-dear-watson/train.csv", nrows = 500)
train.head()
train.premise.values[1]
train.hypothesis.values[1]
train.label.values[1]
labels, frequencies = np.unique(train.language.values, return_counts = True)

plt.figure(figsize = (10,10))
plt.pie(frequencies,labels = labels, autopct = '%1.1f%%')
plt.show()
import numpy as np
import pandas as pd
import time
# Setting TPU if working in Google Colab
%%capture
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version nightly --apt-packages libomp5 libopenblas-dev
%%capture
!pip install pytorch_lightning
!pip install transformers
!pip install nlp
import torch as th
import pytorch_lightning as pl
import nlp
import transformers
from transformers import BertTokenizer
# using basic BERT for sequence classification
LOSS = []
ACC = []
class Sentences(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.brt = transformers.BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels = 3)

    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

        def _tokenize(x):
            print('in tokenize')
            return tokenizer(
                    x['premise'],
                    x['hypothesis'],
                    max_length=64, 
                    truncation = True,
                    pad_to_max_length=True)
            
        def _prepare_ds():
            df = pd.read_csv('../input/contradictory-my-dear-watson/train.csv')
            df_filtered  = df.filter(['premise','hypothesis','label'])
            dataset = nlp.Dataset.from_pandas(df_filtered)
            ds_flt = dataset.train_test_split(test_size=0.1)

            ds_flt['train'] = ds_flt['train'].map(_tokenize, batched=True)
            ds_flt['train'].set_format(type='torch',columns = ['input_ids','token_type_ids','label','attention_mask'])

            ds_flt['test'] = ds_flt['test'].map(_tokenize, batched=True)
            ds_flt['test'].set_format(type='torch',columns = ['input_ids','token_type_ids','label','attention_mask'])

            return ds_flt['train'], ds_flt['test']

        self.train_ds, self.test_ds = _prepare_ds()      

    def forward(self, input_ids, masks, token_type_ids, labels):
        out = self.brt(input_ids, masks, token_type_ids, labels = labels)
        #print('forward brt output - {}'.format(out))
        return out

    def training_step(self, batch, batch_idx):
        res = self.forward(batch['input_ids'],batch['attention_mask'], batch['token_type_ids'], batch['label'])
        LOSS.append(res[0].tolist())
        training_loss = {'train_loss': res[0]}
        return {'loss': res[0], 'log': training_loss} 

    def validation_step(self, batch, batch_idx):
        res = self.forward(batch['input_ids'], batch['attention_mask'], batch['token_type_ids'], batch['label'])
        #print('validation step input - {}'.format(res))
        loss = res[0]
        acc = (res[1].argmax(-1) == batch['label']).float() # argmax(1) or argmax(dim=1) produces the same result
        ACC.append(th.mean(acc))
        out = {'val_loss': loss, 'val_acc': acc}
        print('validation step val_loss & val_acc - {}'.format(out))
        result = pl.EvalResult(checkpoint_on=loss)
        result.log_dict({'val_acc': acc, 'val_loss': loss})
        #return result
        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        #loss = th.cat([o['loss'] for o in outputs], 0).mean()
        for item in outputs:
          loss = item['loss']
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        out = {'val_loss': loss, 'val_acc': acc}
        print('in validation epoc end: out: {}'.format(out)) 
        return {**out, 'log': out}

    def train_dataloader(self):
        return th.utils.data.DataLoader(
                self.train_ds,
                batch_size=32,
                drop_last=True,
                shuffle=True,
                )

    def val_dataloader(self):
        return th.utils.data.DataLoader(
                self.test_ds,
                batch_size=16,
                drop_last=False,
                shuffle=False,
                )

    def configure_optimizers(self):
        return th.optim.SGD(
            self.parameters(),
            lr=0.01,
            momentum=0.9,
        )
def model_fit():
    model = Sentences()
    trainer = pl.Trainer(
        #default_root_dir='logs',
        gpus=(1 if th.cuda.is_available() else 0),
        #tpu_cores = 1, #uncomment if using TPU on Colab
        max_epochs=10,
        fast_dev_run=False,
        logger=pl.loggers.TensorBoardLogger('logs_bert/', name='pretrained'),
    )
    trainer.fit(model)

!rm -rf ./logs_bert/ # these are tensorboard logs

start_time = time.time()
model_fit()
execution_time = time.time() - start_time
#the Tensorboard is set for Colab, I haven't tried it in Kaggle
%load_ext tensorboard
%tensorboard --logdir logs_bert/
predictions = [np.argmax(i) for i in model.predict(test_input)]
submission = test.id.copy().to_frame()
submission['prediction'] = predictions
submission.head()
submission.to_csv("submission.csv", index = False)