# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from scipy.io import loadmat

from glob import glob

from scipy import stats



idx_ = []  # index

len_ = []  # length

mean_ = []  # mean

std_ = []  # standard deviation

ste_ = []  # standard error

max_ = []  # maximum value

min_ = []  # minimum value

var_ = []  # variance

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

        var_.append(x.var())

        if d == 'normal':

            y_.append(0)

        else:

            y_.append(1)
train_df = pd.DataFrame(index=idx_, columns=['length', 'mean', 'standard deviation', 'standard error', 'maximum value', 'minimum value','variance' ,'y'])

train_df['length'] = len_

train_df['mean'] = mean_

train_df['standard deviation'] = std_

train_df['standard error'] = ste_

train_df['maximum value'] = max_

train_df['minimum value'] = min_

train_df['variance'] = var_

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

var_ = []

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

    var_.append(x.var())
test_df = pd.DataFrame(index=idx_, columns=['length', 'mean', 'standard deviation', 'standard error', 'maximum value', 'minimum value','variance'])

test_df['length'] = len_

test_df['mean'] = mean_

test_df['standard deviation'] = std_

test_df['standard error'] = ste_

test_df['maximum value'] = max_

test_df['minimum value'] = min_

test_df['variance'] = var_

test_df
X_train = train_df.drop('y', axis=1).to_numpy()

y_train = train_df['y'].to_numpy()
print(X_train.shape)
#SMOTEで学習用データ作成



from imblearn.over_sampling import SMOTE



# クラス1の数を保存

count_train_class_one = y_train.sum()

print('クラス1のサンプル数:{}'.format(count_train_class_one)) #クラス1のサンプル数表示



# クラス0：クラス1=2:1になるまでクラス1を増やす

smote = SMOTE(sampling_strategy = 0.5, random_state=100)



# 学習用データに反映

X_train_, y_train_ = smote.fit_sample(X_train, y_train)



print(X_train_.shape) #学習用データのサンプル数確認

print("SMOTE後のクラス1のサンプル数:{}".format(y_train_.sum())) #クラス1のサンプル数

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

model.fit(X_train_, y_train_)
p_train = model.predict_proba(X_train)

p_train
y_train_
X_test = test_df.to_numpy()



p_test = model.predict_proba(X_test)

p_test
submit_df = pd.read_csv('/kaggle/input/1056lab-cardiac-arrhythmia-detection/sampleSubmission.csv', index_col=0)

submit_df['af'] = p_test[:,1]

submit_df
submit_df.to_csv('submission.csv')