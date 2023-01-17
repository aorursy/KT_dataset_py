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

from sklearn import preprocessing

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score 

from sklearn.ensemble import RandomForestClassifier

import xgboost

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

import warnings

np.random.seed(123)

warnings.filterwarnings('ignore')

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV
train_dataset = pd.read_csv("/kaggle/input/steam-reviews/train.csv", delimiter=",")

train_dataset
test_dataset = pd.read_csv("/kaggle/input/steam-reviews-test-dataset/test.csv", delimiter=",")

test_dataset['user_suggestion'] = None
dataset = pd.concat([train_dataset, test_dataset], axis = 0)

dataset.reset_index(drop = True, inplace = True)

dataset
from matplotlib import pyplot as plt

import seaborn as sns
# Visualizing the variable - 'year'

plt.figure(figsize = (10,5))

plt.xticks(rotation=90)

sns.countplot(train_dataset['year'])
from fastai.text import *
from pathlib import Path
path = Path('/kaggle/input/steam-reviews/')
%%time

data_lm = TextLMDataBunch.from_csv(path,'train.csv', text_cols = 3, label_cols = 4)
%%time

data_clas = TextClasDataBunch.from_csv(path, 'train.csv', vocab=data_lm.train_ds.vocab, bs=32, text_cols = 3, label_cols = 4)
data_clas
bs=16
#torch.cuda.set_device(1)

#torch.cuda.set_device(0)
learn = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.5)
learn.fit_one_cycle(1, 1e-2)
learn.unfreeze()

learn.fit_one_cycle(3, slice(1e-4,1e-2))
learn.predict("This is a review about", n_words=10)
learn.predict("This game is one of the ", n_words=10)
learn.model_dir = Path('/kaggle/working/')
learn.save(file = Path('language_model'))

learn.save_encoder(Path('language_model_encoder'))
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5).to_fp16()

learn.model_dir = Path('/kaggle/working/')

learn.load_encoder('language_model_encoder')
data_clas.show_batch()
learn.fit_one_cycle(1, 5e-2)
learn.unfreeze()

learn.fit_one_cycle(3, slice(1e-4, 1e-2))
learn.predict('This game is absolute shit! Dont waste your money!')
learn.predict('This is one of the best games to buy!')
learn.predict('The best button of the game is exit button')
#learn.unfreeze()

#learn.fit_one_cycle(3, slice(1e-4, 1e-2))
data_clas.save('/kaggle/working/data_textlist_class')
learn.model_dir = Path('/kaggle/working/')

learn.save('data_model')
data_clas = load_data(path, '/kaggle/working/data_textlist_class', bs=bs, num_workers=1)
data_clas
data_clas = load_data(path, '/kaggle/working/data_textlist_class', bs=bs, num_workers=1)

learn_c = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5, metrics = [accuracy]).to_fp16()

learn_c.model_dir = Path('/kaggle/working/')

learn_c.load('data_model', purge=False);
data_clas_bwd = load_data(path, '/kaggle/working/data_textlist_class', bs=bs, num_workers=1, backwards=True)

learn_c_bwd = text_classifier_learner(data_clas_bwd, AWD_LSTM, drop_mult=0.5, metrics=[accuracy]).to_fp16()
data_clas_bwd.show_batch()
learn_c_bwd.fit_one_cycle(1, 1e-2)
learn_c_bwd.unfreeze()

learn_c_bwd.fit_one_cycle(5, slice(1e-4, 1e-2))
preds,targs = learn_c.get_preds(ordered=True)

accuracy(preds,targs)
preds_b,targs_b = learn_c_bwd.get_preds(ordered=True)

accuracy(preds_b,targs_b)
preds_avg = (preds+preds_b)/2

accuracy(preds_avg,targs_b)