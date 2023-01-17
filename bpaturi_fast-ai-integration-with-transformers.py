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
!pip install --upgrade transformers
!pip install nlp
import numpy as np
import pandas as pd

import nlp
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import fastai
from fastai.text import *
from fastai.metrics import *
import transformers
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification
print("transformers version: ", transformers.__version__)
print("fast.ai version: ", fastai.__version__)
# Creating a config object to store task specific information
class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)
        
config = Config(
    task = "news",
    testing=False,
    seed = 2019,
    roberta_model_name='roberta-large', # can also be exchanged with roberta-base 
    max_lr=1e-5,
    epochs=4,
    use_fp16=False,
    bs=4, 
    max_seq_len=512, 
    num_labels = 2,
    hidden_dropout_prob=.05,
    hidden_size=768, # 1024 for roberta-large
    start_tok = "<s>",
    end_tok = "</s>",
    mark_fields=True)
class FastAiRobertaTokenizer(BaseTokenizer):
    def __init__(self, tokenizer: RobertaTokenizer, max_seq_len: int=128, **kwargs): 
        self._pretrained_tokenizer = tokenizer
        self.max_seq_len = max_seq_len 
    def __call__(self, *args, **kwargs): 
        return self 
    def tokenizer(self, t:str) -> List[str]: 
        return ["<s>"] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + ["</s>"]
    
    
class RobertaTokenizeProcessor(TokenizeProcessor):
    def __init__(self, tokenizer):
        super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)
         
class RobertaNumericalizeProcessor(NumericalizeProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, vocab=fastai_vocab, **kwargs)
        
def get_roberta_processor(tokenizer:Tokenizer=None, vocab:Vocab=None):
    return [RobertaTokenizeProcessor(tokenizer=tokenizer), NumericalizeProcessor(vocab=vocab)]


class RobertaDataBunch(TextDataBunch):
    @classmethod
    def create(cls, train_ds, valid_ds, test_ds=None, path:PathOrStr='.', bs:int=64, val_bs:int=None, pad_idx=1,
               pad_first=True, device:torch.device=None, no_check:bool=False, backwards:bool=False, 
               dl_tfms:Optional[Collection[Callable]]=None, **dl_kwargs) -> DataBunch:
        
        "Function that transform the `datasets` in a `DataBunch` for classification. Passes `**dl_kwargs` on to `DataLoader()`"
        datasets = cls._init_ds(train_ds, valid_ds, test_ds)
        val_bs = ifnone(val_bs, bs)
        collate_fn = partial(pad_collate, pad_idx=pad_idx, pad_first=pad_first, backwards=backwards)
        train_sampler = SortishSampler(datasets[0].x, key=lambda t: len(datasets[0][t][0].data), bs=bs)
        train_dl = DataLoader(datasets[0], batch_size=bs, sampler=train_sampler, drop_last=True, **dl_kwargs)
        dataloaders = [train_dl]
        for ds in datasets[1:]:
            lengths = [len(t) for t in ds.x.items]
            sampler = SortSampler(ds.x, key=lengths.__getitem__)
            dataloaders.append(DataLoader(ds, batch_size=val_bs, sampler=sampler, **dl_kwargs))
        return cls(*dataloaders, path=path, device=device, dl_tfms=dl_tfms, collate_fn=collate_fn, no_check=no_check)
    

class RobertaTextList(TextList):
    _bunch = RobertaDataBunch
    _label_cls = TextList
dataset = nlp.load_dataset('boolq')

''' Prepare pandas DF'''
feat_cols = "review"
label_cols = "answer"

train_data = pd.DataFrame(dataset["train"])
test_data = pd.DataFrame(dataset["validation"])


train_data["review"] = train_data["passage"] + " </s> </s> " + train_data["question"]
test_data["review"] = test_data["passage"] + " </s> </s> " + test_data["question"]

train_data[label_cols] = train_data[label_cols].astype(int)
test_data[label_cols] = test_data[label_cols].astype(int)

combined_df = pd.concat([train_data, test_data])

X_train, X_val, y_train, y_val = train_test_split(combined_df[feat_cols], combined_df[label_cols], \
                                                  test_size=0.2, random_state=101)

train_data = pd.concat([X_train, y_train], axis = 1)
val_data = pd.concat([X_val, y_val], axis = 1)

X_train, X_val, y_train, y_val = train_test_split(val_data[feat_cols], val_data[label_cols], \
                                                  test_size=0.5, random_state=10)

val_data = pd.concat([X_train, y_train], axis = 1)
test_data = pd.concat([X_val, y_val], axis = 1)


print("train shape: ", len(train_data))
print("val shape: ", len(val_data))
print("test shape: ", len(test_data))
train_data.iloc[1]["review"]
''' Initialize Roberta tokenizer'''
roberta_tok = RobertaTokenizer.from_pretrained(config.roberta_model_name)

fastai_tokenizer = Tokenizer(tok_func = FastAiRobertaTokenizer(roberta_tok, \
                                                               max_seq_len=config.max_seq_len), \
                             pre_rules=[], post_rules=[])

''' Construct fast.ai vocab using RobertaTokenizer dictionary'''
path = Path()
roberta_tok.save_vocabulary(path)
with open('vocab.json', 'r') as f:
    roberta_vocab_dict = json.load(f)
    
fastai_roberta_vocab = Vocab(list(roberta_vocab_dict.keys()))
print("Roberta dict size: ", len(roberta_vocab_dict.keys()))


print("Batch size is : ", config.bs)
# loading the tokenizer and vocab processors
processor = get_roberta_processor(tokenizer=fastai_tokenizer, vocab=fastai_roberta_vocab)

# creating our databunch 
data = ItemLists(".", RobertaTextList.from_df(train_data, ".", cols=feat_cols, processor=processor),
                      RobertaTextList.from_df(val_data, ".", cols=feat_cols, processor=processor)
                ) \
       .label_from_df(cols=label_cols, label_cls=CategoryList) \
       .add_test(RobertaTextList.from_df(test_data, ".", cols=feat_cols, processor=processor)) \
       .databunch(bs=config.bs,pad_first=False)
# defining our model architecture 
class RobertaForSequenceClassificationModel(nn.Module):
    def __init__(self,num_labels=config.num_labels):
        super(RobertaForSequenceClassificationModel,self).__init__()
        self.num_labels = num_labels
        self.roberta = RobertaForSequenceClassification.from_pretrained(config.roberta_model_name,num_labels= self.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        outputs = self.roberta(input_ids, token_type_ids, attention_mask)
        logits = outputs[0] 
        return logits
roberta_model = RobertaForSequenceClassificationModel(config.num_labels) 
learn = Learner(data, roberta_model, metrics=[accuracy])
learn.model.roberta.train() # setting roberta to train as it is in eval mode by default
learn.lr_find()
learn.recorder.plot()
print("Cuda available: ", torch.cuda.is_available())
# Looks like 2 epochs are enough on Large transformers
learn.fit_one_cycle(3, max_lr=1e-5)
def get_preds_as_nparray(ds_type, p_learn) -> np.ndarray:
    p_learn.model.roberta.eval()
    preds = p_learn.get_preds(ds_type)[0].detach().cpu().numpy()
    
    sampler = [i for i in data.dl(ds_type).sampler]
    reverse_sampler = np.argsort(sampler)
    ordered_preds = preds[reverse_sampler, :]
    pred_values = np.argmax(ordered_preds, axis=1)
    return ordered_preds, pred_values
# val preds
preds, pred_values = get_preds_as_nparray(DatasetType.Valid, learn)
acc = (pred_values == data.valid_ds.y.items).mean()
print("Validation accuracy: ", acc)
num_samples = 0
for idx, row in val_data.reset_index(drop=True).iterrows():
    if num_samples < 60:
        print(row["review"])
        print("Ground truth: ", row["answer"])
        print("Prediction: ", pred_values[idx])
        print("\n\n")
        num_samples += 1
    else:
        break
learn.save("roberta_large_tuned_boolq_model")
model = RobertaForSequenceClassificationModel(config.num_labels) 
new_learner = Learner(data, model, metrics=[accuracy])
new_learner.model.roberta.eval()
new_learner.load("roberta_large_tuned_boolq_model")

preds, pred_values = get_preds_as_nparray(DatasetType.Valid, new_learner)
(pred_values == data.valid_ds.y.items).mean()
class_map = {0: "False", 1: "True"}
l_str = """<s> Reliance is charged with criminal offence due to insider trading </s> </s> 
is reliance not charged with some offence </s>"""
print("prediction: ", class_map[torch.argmax(new_learner.predict(l_str)[2]).item()])
l_str = """<s> Reliance is charged with criminal offence due to insider trading </s> </s> 
is reliance charged with some offence </s>"""
print("prediction: ", class_map[torch.argmax(new_learner.predict(l_str)[2]).item()])
l_str = """<s> Besides a case against Malaysian tycoon and Maxis owner T. Ananda Krishnan, 
Maxis senior executive Ralph Marshall, former Telecom Minister Dayanidhi Maran and three companies in 
connection with the controversial Aircel Maxis deal, India\'s Central Bureau of Investigations (CBI) has also 
booked Maran\'s brother Kalanidhi, and three companies, Aspro, Maxis and Sun TV, in the case on charges of 
criminal conspiracy under IPC and Prevention of Corruption Act. CBI has registered a case against Maran brothers, Ralph Marshall and T Anandkrishnan and 
three companies under section 120b of IPC read with 13(2) with 13 (1)(d) and also section 7 and 12 of the 
Prevention of Corruption Act. </s> </s>
is Sun Tv not committed any offence like corruption </s>
"""
print("probs: ", new_learner.predict(l_str)[2])
print("prediction: ", class_map[torch.argmax(new_learner.predict(l_str)[2]).item()])
l_str = """ <s> mBanks compliance policy has a special focus on the following issues: prevention of money laundering 
and terrorist financing; appropriate handling of confidential information; 
protection of personal data; supervision of legal compliance in the brokerage and custody business; 
avoidance of conflicts of interest; compliance with rules of giving and accepting gifts by Bank executive 
officers and employees; verification of legal compliance under outsourcing agreements signed by the Bank; 
obligatory publication and reporting to relevant regulators on events in the operation of the Bank; 
advisory to organisational units of the Bank on compliance with new and existing legislation and 
market standards. </s> </s>
is mBank committed any crime like terrorist financing </s>
"""
print("prediction: ", class_map[torch.argmax(new_learner.predict(l_str)[2]).item()])
 
l_str = """ <s> In 2009, Japanese regulators again took action against Citibank Japan, 
because the Bank of Nova Scotia (Scotiabank) had not set up an effective money laundering 
monitoring system. </s> </s>
is Scotiabank committed any offence </s>
"""
print("prediction: ", class_map[torch.argmax(new_learner.predict(l_str)[2]).item()])
