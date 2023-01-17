!pip install -q transformers

!wget https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar

!tar -xvf ./scibert_scivocab_uncased.tar
!pip install -q langdetect

!pip install -q googletrans

from googletrans import Translator

from langdetect import detect
import os

os.environ["WANDB_API_KEY"] = "0" ## to silence warning

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd



import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 12)

import seaborn as sns



import torch

import torch.optim as optim

import torch.nn as nn

import random 

from sklearn.metrics import f1_score,accuracy_score,roc_auc_score



# fastai

import fastai

from fastai import *

from fastai.text import *

from fastai.callbacks import *



# transformers

import transformers

from transformers import *

from transformers import PreTrainedModel, PreTrainedTokenizer, PretrainedConfig



from fastai.callbacks import *

from transformers import AdamW

from functools import partial



import re



print('fastai version :', fastai.__version__)

print('transformers version :', transformers.__version__)
train_data = pd.read_csv('/kaggle/input/researchtopictags/train.csv')

test_data = pd.read_csv('/kaggle/input/researchtopictags/test.csv')  
print(train_data.shape)

train_data.head()
print(test_data.shape)

test_data.head()
train_data['combined_text'] = train_data['TITLE'] + " <join> " + train_data['ABSTRACT']

test_data['combined_text'] = test_data['TITLE'] + " <join> " + test_data['ABSTRACT']
topics = ['Computer Science','Physics','Mathematics', 'Statistics','Quantitative Biology','Quantitative Finance']
com_sc = train_data['Computer Science'].value_counts()[1]

phy = train_data['Physics'].value_counts()[1]

mat = train_data['Mathematics'].value_counts()[1]

stats = train_data['Statistics'].value_counts()[1]

bio = train_data['Quantitative Biology'].value_counts()[1]

fin = train_data['Quantitative Finance'].value_counts()[1]



fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

counts = [com_sc,phy,mat,stats,bio,fin]

ax.bar(topics,counts)

plt.show()
train_data['length'] = train_data['combined_text'].apply(lambda x : x.count(" ") + 1)

sns.distplot(train_data['length'])
SciBertTokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')



SciBertModel = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
MODEL_CLASSES = {'scibert' : (SciBertModel, SciBertTokenizer, PretrainedConfig.from_json_file('./scibert_scivocab_uncased/config.json'))}
# Parameters



seed = 42



use_fp16 = False



bs = 16



threshold = 0.4



MAX_LEN = 320



model_type = 'scibert'



pretrained_model_name = 'allenai/scibert_scivocab_uncased'
model_class, tokenizer_class, config_class = MODEL_CLASSES[model_type]
def seed_all(seed_value):

    

    random.seed(seed_value) # Python

    

    np.random.seed(seed_value) # cpu vars

    

    torch.manual_seed(seed_value) # cpu  vars

    

    if torch.cuda.is_available(): 

        

        torch.cuda.manual_seed(seed_value)

        

        torch.cuda.manual_seed_all(seed_value) # gpu vars

        

        torch.backends.cudnn.deterministic = True  #needed

        

        torch.backends.cudnn.benchmark = False
seed_all(seed)
class TransformersBaseTokenizer(BaseTokenizer):

    

    """Wrapper around PreTrainedTokenizer to be compatible with fast.ai"""

    

    def __init__(self, pretrained_tokenizer: PreTrainedTokenizer, model_type = 'bert', max_len = MAX_LEN,**kwargs):

        

        self._pretrained_tokenizer = pretrained_tokenizer

        

        self.max_seq_len = max_len

        

        self.model_type = model_type



    def __call__(self, *args, **kwargs): 

        

        return self



    def tokenizer(self, t:str) -> List[str]:

        

        """Limits the maximum sequence length and add the special tokens"""

        

        CLS = self._pretrained_tokenizer.cls_token

        

        SEP = self._pretrained_tokenizer.sep_token

        

        tokens = self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2]

        

        tokens = [CLS] + tokens + [SEP]

        

        return tokens
transformer_tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)



transformer_base_tokenizer = TransformersBaseTokenizer(pretrained_tokenizer = transformer_tokenizer, model_type = model_type)



fastai_tokenizer = Tokenizer(tok_func = transformer_base_tokenizer, pre_rules=[], post_rules=[])
class TransformersVocab(Vocab):

    

    def __init__(self, tokenizer: PreTrainedTokenizer):

        

        super(TransformersVocab, self).__init__(itos = [])

        

        self.tokenizer = tokenizer

    

    def numericalize(self, t:Collection[str]) -> List[int]:

        

        "Convert a list of tokens `t` to their ids."

        

        return self.tokenizer.convert_tokens_to_ids(t)



    def textify(self, nums:Collection[int], sep=' ') -> List[str]:

        

        "Convert a list of `nums` to their tokens."

        

        nums = np.array(nums).tolist()

        

        return sep.join(self.tokenizer.convert_ids_to_tokens(nums)) if sep is not None else self.tokenizer.convert_ids_to_tokens(nums)

    

    def __getstate__(self):

        

        return {'itos':self.itos, 'tokenizer':self.tokenizer}



    def __setstate__(self, state:dict):

        

        self.itos = state['itos']

        

        self.tokenizer = state['tokenizer']

        

        self.stoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.itos)})
transformer_vocab =  TransformersVocab(tokenizer = transformer_tokenizer)



numericalize_processor = NumericalizeProcessor(vocab=transformer_vocab)



tokenize_processor = TokenizeProcessor(tokenizer=fastai_tokenizer, include_bos=False, include_eos=False)



transformer_processor = [tokenize_processor, numericalize_processor]
pad_first = False



pad_idx = transformer_tokenizer.pad_token_id
data_classifier = (TextList.from_df(train_data, cols='combined_text', processor=transformer_processor)

                         .split_by_rand_pct(0.3,seed=seed)

                         .label_from_df(cols= topics)

                         .add_test(test_data)

                         .databunch(bs=bs, pad_first=pad_first, pad_idx=pad_idx))
data_classifier.save('data_classifier.pkl')
data_classifier = load_data('','data_classifier.pkl',bs=bs)
print('[CLS] token :', transformer_tokenizer.cls_token)



print('[SEP] token :', transformer_tokenizer.sep_token)



print('[PAD] token :', transformer_tokenizer.pad_token)



data_classifier.show_batch()
print('[CLS] id :', transformer_tokenizer.cls_token_id)



print('[SEP] id :', transformer_tokenizer.sep_token_id)



print('[PAD] id :', pad_idx)



test_one_batch = data_classifier.one_batch()[0]



print('Batch shape : ',test_one_batch.shape)



print(test_one_batch)
# defining our model architecture 



class CustomTransformerModel(nn.Module):

    

    def __init__(self, transformer_model: PreTrainedModel):

        

        super(CustomTransformerModel,self).__init__()

        

        self.transformer = transformer_model

        

        self.classifier = nn.Sequential(#nn.Linear(in_features=768, out_features=768, bias=True),

                                        #nn.Dropout(p=0.1, inplace=False),

                                        nn.Linear(in_features=768, out_features=6, bias=True))

        

    def forward(self, input_ids, attention_mask=None):

        

   

        outputs = self.transformer(input_ids,

                                  attention_mask = attention_mask)

        

        pooled_output = outputs[1]

        

        logits = self.classifier(pooled_output)

        

        return logits
config = config_class.from_pretrained(pretrained_model_name)



config.num_labels = 6



config.use_bfloat16 = use_fp16



print(config)
transformer_model = model_class.from_pretrained(pretrained_model_name, config = config)



custom_transformer_model = CustomTransformerModel(transformer_model = transformer_model)
class MicroF1(Callback):



    _order = -20 #is crucial - without it the custom columns will not be added - it tells the callback system to run this callback before the recorder system.



    def __init__(self,learn,thresh,eps = 1e-15, sigmoid = True,**kwargs):

        

        self.learn = learn

        

        self.thresh = thresh

        

        self.eps = eps

        

        self.sigmoid = sigmoid



    def on_train_begin(self, **kwargs): 

        

        self.learn.recorder.add_metric_names(['MicroF1'])

    

    def on_epoch_begin(self, **kwargs):

        

        self.tp = 0

        

        self.total_pred = 0

        

        self.total_targ = 0

    

    def on_batch_end(self, last_output, last_target, **kwargs):

        

        pred, targ = ((last_output.sigmoid() if self.sigmoid else last_output) > self.thresh).byte(), last_target.byte()

        

        if torch.equal(torch.tensor(pred.shape),torch.tensor(targ.shape)):

            

            m = pred*targ

            

            self.tp += m.sum(0).float()

            

            self.total_pred += pred.sum(0).float()

            

            self.total_targ += targ.sum(0).float()

    

    def fbeta_score(self, precision, recall):

        

        return 2*(precision*recall)/((precision + recall) + self.eps)



    def on_epoch_end(self, last_metrics, **kwargs):

        

        self.total_pred += self.eps

        

        self.total_targ += self.eps

        

        precision, recall = self.tp.sum() / self.total_pred.sum(), self.tp.sum() / self.total_targ.sum()

        

        res = self.fbeta_score(precision, recall)        

        

        return add_metrics(last_metrics, res)
class AUCROC(Callback):

    

    _order = -20 

    

    def __init__(self, learn, **kwargs): 

        

        self.learn = learn

        

        self.output, self.target = [], []

        

    def on_train_begin(self, **kwargs): 

        

        self.learn.recorder.add_metric_names(topics)

        

    def on_epoch_begin(self, **kwargs): 

        

        self.output, self.target = [], []

    

    def on_batch_end(self, last_target, last_output, train, **kwargs):

        

        if not train:

            

            self.output.append(last_output)

            

            self.target.append(last_target)

                

    def on_epoch_end(self, last_metrics, **kwargs):

        

        if len(self.output) > 0:

            

            output = torch.cat(self.output)

            

            target = torch.cat(self.target)

            

            preds = F.softmax(output, dim=1)

            

            metric = []



            for i in range(0,target.shape[1]):

                

                

                metric.append(roc_auc_score(target.cpu().numpy()[...,i], preds[...,i].cpu().numpy(),average='macro'))

            

            return add_metrics(last_metrics, metric)

        

        else:

            

            return
microF1 = partial(MicroF1,thresh = threshold) #metric



CustomAdamW = partial(AdamW, correct_bias=False) #optimizer
classifierModel = Learner(data_classifier, 

                  custom_transformer_model, 

                  opt_func = CustomAdamW, 

                  callback_fns = [microF1,AUCROC],

                  loss_func = nn.BCEWithLogitsLoss()

                 )



# Show graph of learner stats and metrics after each epoch.

classifierModel.callbacks.append(ShowGraph(classifierModel))
print(classifierModel.model)
#n = len(classifierModel.model.transformer.base_model.encoder.layer)//3



#list_layers = [[classifierModel.model.transformer.base_model.embeddings],

#               list(classifierModel.model.transformer.base_model.encoder.layer[:n]),

#               list(classifierModel.model.transformer.base_model.encoder.layer[n+1:2*n]),

#               list(classifierModel.model.transformer.base_model.encoder.layer[(2*n)+1:]),

#               [classifierModel.model.transformer.base_model.pooler],

#               classifierModel.model.classifier]
list_layers = [classifierModel.model.transformer.base_model.embeddings,

              classifierModel.model.transformer.base_model.encoder.layer[0],

              classifierModel.model.transformer.base_model.encoder.layer[1],

              classifierModel.model.transformer.base_model.encoder.layer[2],

              classifierModel.model.transformer.base_model.encoder.layer[3],

              classifierModel.model.transformer.base_model.encoder.layer[4],

              classifierModel.model.transformer.base_model.encoder.layer[5],

              classifierModel.model.transformer.base_model.encoder.layer[6],

              classifierModel.model.transformer.base_model.encoder.layer[7],

              classifierModel.model.transformer.base_model.encoder.layer[8],

              classifierModel.model.transformer.base_model.encoder.layer[9],

              classifierModel.model.transformer.base_model.encoder.layer[10],

              classifierModel.model.transformer.base_model.encoder.layer[11],

              classifierModel.model.transformer.base_model.pooler]
classifierModel.split(list_layers)



num_groups = len(classifierModel.layer_groups)



print('Learner split in',num_groups,'groups')



print(classifierModel.layer_groups)

classifierModel.save('untrain')
seed_all(seed)



classifierModel.load('untrain');
classifierModel.freeze_to(-1)
classifierModel.summary()
classifierModel.lr_find()
classifierModel.recorder.plot(skip_end=10,suggestion=True)
classifierModel.fit_one_cycle(5, max_lr = 3e-3 ,moms=(0.8,0.7))
classifierModel.save('classifierModel1')
seed_all(seed)



classifierModel.load('classifierModel1');
for i in range(2,6):

    

    print('=' * 50, f" Frozen Layer Group {i}", '=' * 50)

    

    classifierModel.freeze_to(-i)

    

    classifierModel.fit_one_cycle(1,slice(1e-6,2e-6),moms=(0.8,0.7))

    

    print ('')
classifierModel.save('classifierModel2')
seed_all(seed)



classifierModel.load('classifierModel2');
classifierModel.unfreeze()
print('=' * 50, f" All Layers Unfrozen ", '=' * 50)

classifierModel.fit_one_cycle(2, max_lr=slice(1e-6,2e-6), moms=(0.8, 0.9))
classifierModel.show_results()
class_probs = classifierModel.get_preds(DatasetType.Test)[0]
def get_preds_as_nparray(ds_type) -> np.ndarray:

    """

    the get_preds method does not yield the elements in order by default

    we borrow the code from the RNNLearner to resort the elements into their correct order

    """

    preds = (class_probs > 0.4).byte().detach().cpu().numpy()

    

    sampler = [i for i in data_classifier.dl(ds_type).sampler]

    

    reverse_sampler = np.argsort(sampler)

    

    return preds[reverse_sampler, :]



preds = get_preds_as_nparray(DatasetType.Test)
submission = pd.read_csv('/kaggle/input/researchtopictags/sample.csv')



submission.iloc[:,1:] =  preds



submission.to_csv('submission.csv', index=False)



submission.head()