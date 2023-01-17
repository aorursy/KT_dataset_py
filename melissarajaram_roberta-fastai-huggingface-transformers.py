!pip install ../input/sacremoses/sacremoses-master 

!pip install ../input/transformers/transformers-master 
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pathlib import Path 



import os



import torch

import torch.optim as optim



import random 



# fastai

from fastai import *

from fastai.text import *

from fastai.callbacks import *



# classification metric

from scipy.stats import spearmanr



# transformers

from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig

from transformers import RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    print(dirname)
def seed_all(seed_value):

    random.seed(seed_value) # Python

    np.random.seed(seed_value) # cpu vars

    torch.manual_seed(seed_value) # cpu  vars

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        torch.backends.cudnn.deterministic = True  #needed

        torch.backends.cudnn.benchmark = False
seed=42

seed_all(seed)
DATA_ROOT = Path("../input/google-quest-challenge/")

MODEL_ROOT = Path("../input/robertabasepretrained")

train = pd.read_csv(DATA_ROOT / 'train.csv')

test = pd.read_csv(DATA_ROOT / 'test.csv')

sample_sub = pd.read_csv(DATA_ROOT / 'sample_submission.csv')

print(train.shape,test.shape)
train.head()
labels = list(sample_sub.columns[1:].values)
for label in labels: print(label) 
MODEL_CLASSES = {

    'roberta': (RobertaForSequenceClassification, RobertaTokenizer, RobertaConfig),

}
# Parameters

seed = 42

use_fp16 = False

bs = 8



model_type = 'roberta'

pretrained_model_name = 'roberta-base' # 'roberta-base-openai-detector'
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
model_class.pretrained_model_archive_map.keys()
class TransformersBaseTokenizer(BaseTokenizer):

    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""

    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', **kwargs):

        self._pretrained_tokenizer = pretrained_tokenizer

        self.max_seq_len = pretrained_tokenizer.max_len

        self.model_type = model_type



    def __call__(self, *args, **kwargs): 

        return self



    def tokenizer(self, t:str) -> List[str]:

        """Limits the maximum sequence length and add the spesial tokens"""

        CLS = self._pretrained_tokenizer.cls_token

        SEP = self._pretrained_tokenizer.sep_token

        if self.model_type in ['roberta']:

            tokens = self._pretrained_tokenizer.tokenize(t, add_prefix_space=True)[:self.max_seq_len - 2]

        else:

            tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]

        return [CLS] + tokens + [SEP]
transformer_tokenizer = tokenizer_class.from_pretrained(MODEL_ROOT)

transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)

fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])
class TransformersVocab(Vocab):

    def __init__(self, tokenizer: PreTrainedTokenizer):

        super(TransformersVocab, self).__init__(itos = [])

        self.tokenizer = tokenizer

    

    def numericalize(self, t:Collection[str]) -> List[int]:

        "Convert a list of tokens `t` to their ids."

        return self.tokenizer.convert_tokens_to_ids(t)

        #return self.tokenizer.encode(t)



    def textify(self, nums:Collection[int], sep=' ') -> List[str]:

        "Convert a list of `nums` to their tokens."

        nums = np.array(nums).tolist()

        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)
transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)

numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)



tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)



transformer_processor = [tokenize_processor, numericalize_processor]
pad_first = bool(model_type in ['xlnet'])

pad_idx = transformer_tokenizer.pad_token_id
databunch = (TextList.from_df(train, cols=['question_title','question_body','answer'], processor=transformer_processor)

             .split_by_rand_pct(0.1,seed=seed)

             .label_from_df(cols=labels)

             .add_test(test)

             .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))
print('[CLS] token :', transformer_tokenizer.cls_token)

print('[SEP] token :', transformer_tokenizer.sep_token)

print('[PAD] token :', transformer_tokenizer.pad_token)

databunch.show_batch()
print('[CLS] id :', transformer_tokenizer.cls_token_id)

print('[SEP] id :', transformer_tokenizer.sep_token_id)

print('[PAD] id :', pad_idx)

test_one_batch = databunch.one_batch()[0]

print('Batch shape : ',test_one_batch.shape)

print(test_one_batch)
# defining our model architecture 

class CustomTransformerModel(nn.Module):

    def __init__(self, transformer_model: PreTrainedModel):

        super(CustomTransformerModel,self).__init__()

        self.transformer = transformer_model

        

    def forward(self, input_ids, attention_mask=None):

        

        attention_mask = (input_ids!=1).type(input_ids.type()) # attention_mask for RoBERTa

            

        logits = self.transformer(input_ids,

                                attention_mask = attention_mask)[0]   

        return logits
config = config_class.from_pretrained(MODEL_ROOT)

transformer_model = model_class.from_pretrained(MODEL_ROOT, config = config)

custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)
class AvgSpearman(Callback):

    

    def on_epoch_begin(self, **kwargs):

        self.preds = np.empty( shape=(0, 30) )

        self.target = np.empty( shape=(0, 30) )

    

    def on_batch_end(self, last_output, last_target, **kwargs):

        self.preds = np.append(self.preds,last_output,axis=0)

        self.target = np.append(self.target,last_target,axis=0)

    

    def on_epoch_end(self, last_metrics, **kwargs):

        spearsum = 0

        for col in range(self.preds.shape[1]):

            spearsum += spearmanr(self.preds[:,col],self.target[:,col]).correlation

        res = spearsum / (self.preds.shape[1] + 1)

        return add_metrics(last_metrics, res)
from fastai.callbacks import *

from transformers import AdamW



learner = Learner(databunch, 

                  custom_transformer_model, 

                  opt_func = lambda input: AdamW(input,correct_bias=False), 

                  metrics=[AvgSpearman()])



# Show graph of learner stats and metrics after each epoch.

learner.callbacks.append(ShowGraph(learner))



# Put learn in FP16 precision mode. --> Not working in the tutorial

if use_fp16: learner = learner.to_fp16()
print(learner.model)
num_groups = len(learner.layer_groups)

print('Learner split in',num_groups,'groups')
list_layers = [learner.model.transformer.roberta.embeddings,

              learner.model.transformer.roberta.encoder.layer[0],

              learner.model.transformer.roberta.encoder.layer[1],

              learner.model.transformer.roberta.encoder.layer[2],

              learner.model.transformer.roberta.encoder.layer[3],

              learner.model.transformer.roberta.encoder.layer[4],

              learner.model.transformer.roberta.encoder.layer[5],

              learner.model.transformer.roberta.encoder.layer[6],

              learner.model.transformer.roberta.encoder.layer[7],

              learner.model.transformer.roberta.encoder.layer[8],

              learner.model.transformer.roberta.encoder.layer[9],

              learner.model.transformer.roberta.encoder.layer[10],

              learner.model.transformer.roberta.encoder.layer[11],

              learner.model.transformer.roberta.pooler]



learner.split(list_layers);
num_groups = len(learner.layer_groups)

print('Learner split in',num_groups,'groups')
seed_all(seed)

learner.freeze_to(-1)
learner.lr_find()
learner.recorder.plot(skip_end=7,suggestion=True)
unfreeze_layers = [-1,-2,-3,-4,-5,-7]

learning_rates = [2e-4, 5e-5, 1e-5, 5e-6, 1e-6,5e-7]

epochs = [6, 5, 5, 5, 7, 7]
def model_unfreezing_and_training():

    for layer in range(0,len(unfreeze_layers)):

        learner.freeze_to(unfreeze_layers[layer])

        print('freezing to:',unfreeze_layers[layer],' - ',epochs[layer],'epochs')

        learner.fit_one_cycle(epochs[layer], 

                              max_lr=slice(learning_rates[layer]*0.95**num_groups, learning_rates[layer]),

                              moms=(0.8, 0.9))

        learner.save('cycle_'+str(layer))
model_unfreezing_and_training()
def get_preds_as_nparray(ds_type) -> np.ndarray:

    """

    the get_preds method does not yield the elements in order by default

    we borrow the code from the RNNLearner to resort the elements into their correct order

    """

    preds = learner.get_preds(ds_type)[0].detach().cpu().numpy()

    sampler = [i for i in databunch.dl(ds_type).sampler]

    reverse_sampler = np.argsort(sampler)

    return preds[reverse_sampler, :]



test_preds = get_preds_as_nparray(DatasetType.Test)
test_preds,test_preds.shape
sample_submission = pd.read_csv(DATA_ROOT / 'sample_submission.csv')

sample_submission[labels] = test_preds

sample_submission.to_csv("submission.csv", index=False)
test.head()
sample_submission.head()