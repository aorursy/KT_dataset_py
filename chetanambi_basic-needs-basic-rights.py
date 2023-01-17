!pip install simpletransformers
import os

os.listdir('/kaggle/input/basic-needs-basic-rights-new/')
import os

import gc

import re

import pandas as pd

import numpy as np

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from sklearn.metrics import accuracy_score, log_loss
train = pd.read_csv('/kaggle/input/basic-needs-basic-rights-new/train_spell_corrected.csv')

test = pd.read_csv('/kaggle/input/basic-needs-basic-rights-new/test_spell_corrected.csv')

sub = pd.read_csv('/kaggle/input/basicneedsbasicneeds/SampleSubmission.csv')
train.shape, test.shape, sub.shape
train.head(3)
train.isna().sum()
train.drop('ID', axis=1, inplace=True)

test.drop('ID', axis=1, inplace=True)
train['label'].value_counts()
test[test.duplicated()]
print(train['text'].apply(lambda x: len(x.split())).describe())
print(test['text'].apply(lambda x: len(x)).describe())
#train['label'] = train['label'].map({'Depression':0, 'Alcohol': 1, 'Suicide': 2, 'Drugs': 3})
train.columns = ['text','labels']
from simpletransformers.classification import ClassificationModel

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from sklearn.metrics import accuracy_score, log_loss

from scipy.special import softmax
model_args = {'train_batch_size': 50, 

              'reprocess_input_data': True,

              'overwrite_output_dir': True,

              'fp16': False,

              'do_lower_case': False,

              'num_train_epochs': 7,

              'max_seq_length': 160,

              'regression': False,

              'manual_seed': 1994,

              "learning_rate": 3e-5,

              'weight_decay': 0,

              "save_eval_checkpoints": False,

              "save_model_every_epoch": False,

              'no_cache':True,

              "silent": True,

              "use_early_stopping": True,

              "early_stopping_delta": 0.01,

              "early_stopping_metric": "mcc",

              "early_stopping_metric_minimize": False,

              "early_stopping_patience": 5,

              "evaluate_during_training_steps": 1000

              }
def get_model():

    model = ClassificationModel('roberta', 'roberta-base', use_cuda=True, num_labels=4, args=model_args)                            

    return model
err=[]

y_pred_tot=[]

i=1



fold=StratifiedKFold(n_splits=20, shuffle=True, random_state=1994)



for train_index, test_index in fold.split(train, train['labels']):

    train1_trn, train1_val = train.iloc[train_index], train.iloc[test_index]

    model = get_model()

    gc.collect()

    model.train_model(train1_trn)

    score, raw_outputs_val, wrong_preds = model.eval_model(train1_val) 

    raw_outputs_val = softmax(raw_outputs_val, axis=1) 

    print('Log_Loss:', log_loss(train1_val['labels'], raw_outputs_val))

    err.append(log_loss(train1_val['labels'], raw_outputs_val))

    predictions, raw_outputs_test = model.predict(test['text'])

    raw_outputs_test = softmax(raw_outputs_test, axis=1) 

    y_pred_tot.append(raw_outputs_test)
np.mean(err, 0)
y_pred = np.mean(y_pred_tot, 0)
sub[['Depression','Alcohol','Suicide','Drugs']] = y_pred

sub.head()
sub.to_csv('Output.csv', index=False)
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,

    filename=filename)

    return HTML(html)



create_download_link(sub)