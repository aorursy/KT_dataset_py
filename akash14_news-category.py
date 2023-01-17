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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

import os

print(os.listdir("../input"))
from fastai.text import *

from fastai.imports import *

from fastai.text import *

from fastai import *
path = Path('/kaggle/input/news-category-dataset/Participants_Data_News_category/')

path.ls()
train = pd.read_excel(path/'Data_Train.xlsx')

test = pd.read_excel(path/'Data_Test.xlsx')

sub = pd.read_excel(path/'Sample_submission.xlsx')
train.shape, test.shape, sub.shape
train.head(2)
test.head(2)
sns.countplot(x='SECTION', data=train)
def random_seed(seed_value):

    import random 

    random.seed(seed_value)  

    import numpy as np

    np.random.seed(seed_value)  

    import torch

    torch.manual_seed(seed_value)  

    

    if torch.cuda.is_available(): 

        torch.cuda.manual_seed(seed_value)

        torch.cuda.manual_seed_all(seed_value)  

        torch.backends.cudnn.deterministic = True   

        torch.backends.cudnn.benchmark = False
from fastai import *

from fastai.text import *
from sklearn.metrics import accuracy_score 

y_pred_totcb = []

from sklearn.model_selection import KFold, RepeatedKFold

fold = KFold(n_splits=2, shuffle=True, random_state=0)

i=1



for train_index, test_index in fold.split(train):

    

    train_df = train.iloc[train_index]

    valid_df = train.iloc[test_index]



    random_seed(1)

    

    data_lm = TextLMDataBunch.from_df(Path(path), train_df, valid_df, test, text_cols=[0], bs=32)

    data_clas = TextClasDataBunch.from_df(Path(path), train_df, valid_df, test, text_cols=[0], label_cols=1, bs=32)

    

    learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.4, model_dir='/tmp/model/')

    learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))

    learn.unfreeze()

    learn.fit_one_cycle(9, 1e-3, moms=(0.8,0.7))

    learn.save_encoder('model_enc')

    

    learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.4, model_dir='/tmp/model/')

    learn.load_encoder('model_enc')

    learn.fit_one_cycle(1, 1e-2, moms=(0.8, 0.7))

    learn.freeze_to(-2)

    learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))

    learn.freeze_to(-3)

    learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))

    learn.unfreeze()

    learn.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.7))

   

    log_preds, test_labels = learn.get_preds(ds_type=DatasetType.Test, ordered=True)

    preds = np.argmax(log_preds, 1)

    y_pred_totcb.append(preds)

    print(f'fold {i} completed')

    i = i+1
df = pd.DataFrame()

for i in range(1):

    col_name = 'SECTION_' + str(i)

    df[col_name] =  y_pred_totcb[i] 
sub = pd.DataFrame()

sub['SECTION'] = df.mode(axis=1)[0]

sub.tail()
sub['SECTION'].value_counts()
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



def create_download_link(df, title = "Download CSV file", filename = "submission.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



create_download_link(sub)