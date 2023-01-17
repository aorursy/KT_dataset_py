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
from scipy.io import loadmat



mat_contents = loadmat('/kaggle/input/1056lab-cardiac-arrhythmia-detection/af/B00015.mat')

mat_contents['val'][0]
import matplotlib.pyplot as plt



y = mat_contents['val'][0]

length = len(y)

x = np.linspace(0, length, length)



plt.style.use('ggplot')

plt.figure()

plt.plot(x, y)

plt.show()
from glob import glob

import scipy.stats

idx_ = []  # index

len_ = []  # length

mean_ = []  # mean

std_ = []  # standard deviation

ste_ = []  # standard error

max_ = []  # maximum value

min_ = []  # minimum value

med_ = [] # median

skew_ = [] #skewness

kurt_ = [] #kurtosis

y_ = []

for d in ['normal', 'af']:

    for path in sorted(glob('/kaggle/input/1056lab-cardiac-arrhythmia-detection/' + d +'/*.mat')):

        filename = path.split('/')[-1]  # e.g. B05821.mat

        i = filename.split('.')[0]  # e.g. B05821

        idx_.append(i)

        mat_contents = loadmat(path)

        x = mat_contents['val'][0]

        len_.append(len(x))

        mean_.append(x.mean())

        std_.append(x.std())

        ste_.append(x.std()/np.sqrt(len(x)))

        max_.append(x.max())

        min_.append(x.min())

        med_.append(np.median(x))

        skew_.append(scipy.stats.skew(x))

        kurt_.append(scipy.stats.kurtosis(x))

        if d == 'normal':

            y_.append(0)

        else:

            y_.append(1)
train_df = pd.DataFrame(index=idx_, columns=['length', 'mean', 'standard deviation', 'standard error', 'maximum value', 'minimum value', 'median','skewness','kurtosis','y'])

train_df['length'] = len_

train_df['mean'] = mean_

train_df['standard deviation'] = std_

train_df['standard error'] = ste_

train_df['maximum value'] = max_

train_df['minimum value'] = min_

train_df['median'] = med_

train_df['skewness'] = skew_

train_df['kurtosis'] = kurt_

train_df['y'] = y_

train_df
from glob import glob



idx_ = []  # index

len_ = []  # length

mean_ = []  # mean

std_ = []  # standard deviation

ste_ = []  # standard error

max_ = []  # maximum value

min_ = []  # minimum value

med_ = [] # median

skew_ = [] #skewness

kurt_ = [] #kurtosis

for path in sorted(glob('/kaggle/input/1056lab-cardiac-arrhythmia-detection/test/*.mat')):

    filename = path.split('/')[-1]  # e.g. B05821.mat

    i = filename.split('.')[0]  # e.g. B05821

    idx_.append(i)

    mat_contents = loadmat(path)

    x = mat_contents['val'][0]

    len_.append(len(x))

    mean_.append(x.mean())

    std_.append(x.std())

    ste_.append(x.std()/np.sqrt(len(x)))

    max_.append(x.max())

    min_.append(x.min())

    med_.append(np.median(x))

    skew_.append(scipy.stats.skew(x))

    kurt_.append(scipy.stats.kurtosis(x))
test_df = pd.DataFrame(index=idx_, columns=['length', 'mean', 'standard deviation', 'standard error', 'maximum value', 'minimum value','median','skewness','kurtosis'])

test_df['length'] = len_

test_df['mean'] = mean_

test_df['standard deviation'] = std_

test_df['standard error'] = ste_

test_df['maximum value'] = max_

test_df['minimum value'] = min_

test_df['median'] = med_

test_df['skewness'] = skew_

test_df['kurtosis'] = kurt_

test_df
import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('ggplot')

plt.figure()

sns.pairplot(train_df)

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns



corr = train_df.corr()



plt.style.use('ggplot')

plt.figure()

sns.heatmap(corr, square=True, annot=True)

plt.show()
train_df_X = train_df.drop('y', axis=1).values

train_df_y = train_df['y'].values
from sklearn.metrics import auc

#from sklearn.model_selection import train_test_split

#X_train, X_valid, y_train, y_valid = train_test_split(train_df_X, train_df_y, test_size=0.2, random_state=0)
# ライブラリ

from imblearn.over_sampling import SMOTE

# 正例の数を保存

positive_count_train = train_df_y.sum()

print('positive count:{}'.format(positive_count_train))



# SMOTEで不正利用の割合を約20%まで増やす

smote = SMOTE(sampling_strategy={1:positive_count_train*2},random_state=0)



# 学習用データに反映

X_train_resampled, y_train_resampled = smote.fit_sample(train_df_X,train_df_y)

print('y_train_resample:\n{}'.format(pd.Series(y_train_resampled).value_counts()))

X_train_resampled
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(random_state=0)

params = {'criterion':('gini', 'entropy'),

          'max_depth':[1, 2, 3, 4 ,5],

          'n_estimators':[50,100,150,200,250],

         'max_features':[1,'auto',None],

          'min_samples_leaf': [1, 2, 4,]

         }

gscv = GridSearchCV(clf, params, cv=5,scoring='roc_auc')

gscv.fit(X_train_resampled, y_train_resampled)

#model.fit(X_train, y_train)
scores = gscv.cv_results_['mean_test_score']

params = gscv.cv_results_['params']

for score, param in zip(scores, params):

  print('%.3f  %r' % (score, param))
print('%.3f  %r' % (gscv.best_score_, gscv.best_params_))
#gscv.fit(train_df_X, train_df_y)
X_test = test_df.values

p_test = gscv.predict_proba(X_test)

p_test
submit_df = pd.read_csv('/kaggle/input/1056lab-cardiac-arrhythmia-detection/sampleSubmission.csv', index_col=0)

submit_df['af'] = p_test[:,1]

submit_df
submit_df.to_csv('submission2.csv')