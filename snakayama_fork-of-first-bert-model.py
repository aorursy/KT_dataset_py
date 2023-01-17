from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import torch.utils.data

import numpy as np

import pandas as pd

from tqdm import tqdm

import os

import warnings

from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam

from pytorch_pretrained_bert import BertConfig



warnings.filterwarnings(action='once')

device = torch.device('cuda')
def convert_lines(example, max_seq_length,tokenizer):

    max_seq_length -=2

    all_tokens = []

    longer = 0

    for text in tqdm(example):

        tokens_a = tokenizer.tokenize(text)

        if len(tokens_a)>max_seq_length:

            tokens_a = tokens_a[:max_seq_length]

            longer += 1

        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))

        all_tokens.append(one_token)

    return np.array(all_tokens)
MAX_SEQUENCE_LENGTH = 220

SEED = 1234

BATCH_SIZE = 32

BERT_MODEL_PATH = '../input/bertpretrainedmodels/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/'
np.random.seed(SEED)

torch.manual_seed(SEED)

torch.cuda.manual_seed(SEED)

torch.backends.cudnn.deterministic = True
bert_config = BertConfig('../input/bertjson/bert_config_new.json')

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH, cache_dir=None,do_lower_case=True)
test_df = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv")

test_df['comment_text'] = test_df['comment_text'].astype(str) 

X_test = convert_lines(test_df["comment_text"].fillna("DUMMY_VALUE"), MAX_SEQUENCE_LENGTH, tokenizer)
model = BertForSequenceClassification(bert_config, num_labels=16)

model.load_state_dict(torch.load("../input/bert2epaux01wwd1e4decay95weachrebasev1/bert_pytorch.bin"))

model.to(device)

for param in model.parameters():

    param.requires_grad = False

model.eval()
test_preds = np.zeros((len(X_test)))

test = torch.utils.data.TensorDataset(torch.tensor(X_test, dtype=torch.long))

test_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=False)

tk0 = tqdm(test_loader)

for i, (x_batch,) in enumerate(tk0):

    pred = model(x_batch.to(device), attention_mask=(x_batch > 0).to(device), labels=None)

    test_preds[i * 32:(i + 1) * 32] = pred[:, 0].detach().cpu().squeeze().numpy()



test_pred = torch.sigmoid(torch.tensor(test_preds)).numpy().ravel()
from IPython.display import HTML

import pandas as pd

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.utils.validation import check_X_y, check_is_fitted

from sklearn.linear_model import LogisticRegression

from scipy import sparse

import re

import string
train_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')

test_df = pd.read_csv('../input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
text = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')

def tokenize(s): return text.sub(r' \1 ', s).split()

length = train_df.shape[0]

Vectorize = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,

               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,

               smooth_idf=1, sublinear_tf=1 )
train = Vectorize.fit_transform(train_df["comment_text"])

test = Vectorize.transform(test_df["comment_text"])
#Target

y = np.where(train_df['target'] >= 0.5, 1, 0)
class NbSvmClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, C=1.0, dual=False, n_jobs=1):

        self.C = C

        self.dual = dual

        self.n_jobs = n_jobs



    def predict(self, x):

        # Verify that model has been fit

        check_is_fitted(self, ['_r', '_clf'])

        return self._clf.predict(x.multiply(self._r))



    def predict_proba(self, x):

        # Verify that model has been fit

        check_is_fitted(self, ['_r', '_clf'])

        return self._clf.predict_proba(x.multiply(self._r))



    def fit(self, x, y):

        y = y

        x, y = check_X_y(x, y, accept_sparse=True)



        def pr(x, y_i, y):

            p = x[y==y_i].sum(0)

            return (p+1) / ((y==y_i).sum()+1)

        

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))

        x_nb = x.multiply(self._r)

        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)

        return self
NbSvm = NbSvmClassifier(C=1.5, dual=True, n_jobs=-1)

NbSvm.fit(train, y)
prediction=NbSvm.predict_proba(test)[:,1]
submission = pd.read_csv("../input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv")

submission['prediction'] = 0.2*prediction + 0.3*test_pred

submission.to_csv('submission.csv', index=False)