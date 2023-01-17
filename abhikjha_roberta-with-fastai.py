import os

__print__ = print

def print(string):

    os.system(f'echo \"{string}\"')

    __print__(string)
! pip install pytorch-transformers
from fastai.text import *

from fastai.metrics import *

from pytorch_transformers import RobertaTokenizer, DistilBertTokenizer
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

    testing=False,

    seed = 2019,

    roberta_model_name='roberta-base', # can also be exchnaged with roberta-large 

    epochs=2,

    use_fp16=False,

    bs=4, 

    max_seq_len=128, 

    num_labels = 2,

    hidden_dropout_prob=.10,

    hidden_size=768, # 1024 for roberta-large

    start_tok = "<s>",

    end_tok = "</s>",

)
df = pd.read_csv("/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv")
df.shape
if config.testing: df = df[:5000]

print(df.shape)
df.head()
feat_cols = "review"

label_cols = "sentiment"
class FastAiRobertaTokenizer(BaseTokenizer):

    """Wrapper around RobertaTokenizer to be compatible with fastai"""

    def __init__(self, tokenizer: RobertaTokenizer, max_seq_len: int=128, **kwargs): 

        self._pretrained_tokenizer = tokenizer

        self.max_seq_len = max_seq_len 

    def __call__(self, *args, **kwargs): 

        return self 

    def tokenizer(self, t:str) -> List[str]: 

        """Adds Roberta bos and eos tokens and limits the maximum sequence length""" 

        return [config.start_tok] + self._pretrained_tokenizer.tokenize(t)[:self.max_seq_len - 2] + [config.end_tok]
# create fastai tokenizer for roberta

roberta_tok = RobertaTokenizer.from_pretrained("roberta-base")



fastai_tokenizer = Tokenizer(tok_func=FastAiRobertaTokenizer(roberta_tok, max_seq_len=config.max_seq_len), 

                             pre_rules=[], post_rules=[])
# create fastai vocabulary for roberta

path = Path()

roberta_tok.save_vocabulary(path)



with open('vocab.json', 'r') as f:

    roberta_vocab_dict = json.load(f)

    

fastai_roberta_vocab = Vocab(list(roberta_vocab_dict.keys()))
# Setting up pre-processors

class RobertaTokenizeProcessor(TokenizeProcessor):

    def __init__(self, tokenizer):

         super().__init__(tokenizer=tokenizer, include_bos=False, include_eos=False)



class RobertaNumericalizeProcessor(NumericalizeProcessor):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, vocab=fastai_roberta_vocab, **kwargs)





def get_roberta_processor(tokenizer:Tokenizer=None, vocab:Vocab=None):

    """

    Constructing preprocessors for Roberta

    We remove sos and eos tokens since we add that ourselves in the tokenizer.

    We also use a custom vocabulary to match the numericalization with the original Roberta model.

    """

    return [RobertaTokenizeProcessor(tokenizer=tokenizer), NumericalizeProcessor(vocab=vocab)]
from sklearn.model_selection import train_test_split
train, val = train_test_split(df, shuffle=True, test_size=0.2, random_state=42)
train.shape, val.shape
databunch_1 = TextDataBunch.from_df(".", train, val, 

                  tokenizer=fastai_tokenizer,

                  vocab=fastai_roberta_vocab,

                  include_bos=False,

                  include_eos=False,

                  text_cols=feat_cols,

                  label_cols=label_cols,

                  bs=4,

                  collate_fn=partial(pad_collate, pad_first=False, pad_idx=0),

             )
import torch

import torch.nn as nn

from pytorch_transformers import RobertaModel



# defining our model architecture 

class CustomRobertaModel(nn.Module):

    def __init__(self,num_labels=2):

        super(CustomRobertaModel,self).__init__()

        self.num_labels = num_labels

        self.roberta = RobertaModel.from_pretrained(config.roberta_model_name)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = nn.Linear(config.hidden_size, num_labels) # defining final output layer

        

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        _ , pooled_output = self.roberta(input_ids, token_type_ids, attention_mask) # 

        logits = self.classifier(pooled_output)        

        return logits
roberta_model = CustomRobertaModel()



learn = Learner(databunch_1, roberta_model, metrics=[accuracy, error_rate])
learn.model.roberta.train() # setting roberta to train as it is in eval mode by default

learn.fit_one_cycle(config.epochs, 1e-5, wd=1e-5)
learn.recorder.plot_losses()
def roberta_clas_split(self) -> List[nn.Module]:



    

    roberta = roberta_model.roberta

    embedder = roberta.embeddings

    pooler = roberta.pooler

    encoder = roberta.encoder

    classifier = [roberta_model.dropout, roberta_model.classifier]

    n = len(encoder.layer)//3

    print(n)

    print(len(encoder.layer))

    groups = [[embedder], list(encoder.layer[:n]), list(encoder.layer[n+1:2*n]), list(encoder.layer[(2*n)+1:]), [pooler], classifier]

    return groups
x = roberta_clas_split(roberta_model)
learner = Learner(databunch_1, roberta_model, metrics=[accuracy, error_rate])
learner.layer_groups
learner.split([x[0], x[2], x[4]])
learner.layer_groups
learner.freeze()

learner.lr_find()

learner.recorder.plot()
learner.model.roberta.train()

learner.fit_one_cycle(config.epochs, max_lr=slice(1e-7,1e-6), wd =(1e-7, 1e-5, 1e-4))

learner.recorder.plot_losses()
learner.unfreeze()

learner.fit_one_cycle(2, max_lr=slice(1e-7,1e-6), wd =(1e-7, 1e-5, 1e-4))

learner.recorder.plot_losses()
def get_preds_as_nparray(ds_type) -> np.ndarray:

    learn.model.roberta.eval()

    preds = learn.get_preds(ds_type)[0].detach().cpu().numpy()

    sampler = [i for i in databunch_1.dl(ds_type).sampler]

    reverse_sampler = np.argsort(sampler)

    ordered_preds = preds[reverse_sampler, :]

    pred_values = np.argmax(ordered_preds, axis=1)

    return ordered_preds, pred_values
preds, pred_values = get_preds_as_nparray(DatasetType.Valid)
# accuracy on valid

(pred_values == databunch_1.valid_ds.y.items).mean()