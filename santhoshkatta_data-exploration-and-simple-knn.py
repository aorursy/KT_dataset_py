# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import random

random.seed(10)
pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
data = pd.read_csv('../input/costa-rican-household-poverty-prediction/train.csv')
columns_del_list = ['idhogar', 'Id','dependency','tamhog','r4t3', 'tamviv', 'r4h1', 'hogar_total','r4h2', 'r4h3', 'r4m1', 'r4m2','r4m3', 'hogar_nin', 'hogar_adul', 'hogar_mayor', 'SQBescolari','SQBage','SQBhogar_total','SQBedjefe','SQBhogar_nin','SQBovercrowding','SQBdependency','SQBmeaned','agesq']
def drop_columns(data):

    data.drop(columns_del_list, axis=1, inplace=True)
#  Dealing with null or na

def replce_na(data):

    data['v2a1'].fillna(0, inplace=True)

    data['v18q1'].fillna(0, inplace=True)

    data['rez_esc'].fillna(0, inplace=True)

    data['meaneduc'].fillna(data['meaneduc'].median(), inplace=True)
def get_cat_cols(data):

    cols = data.columns

    num_cols = data._get_numeric_data().columns

    cat_columns = list(set(cols) - set(num_cols))

    return cat_columns
def encode_data(data):

    data['edjefe'] = data['edjefe'].replace(['yes'], 1)

    data['edjefe'] = data['edjefe'].replace(['no'], 0)

    data['edjefa'] = data['edjefa'].replace(['yes'], 1)

    data['edjefa'] = data['edjefa'].replace(['no'], 0)
drop_columns(data)

replce_na(data)

get_cat_cols(data)

encode_data(data)
# More biased towards value '4'

print("Target 1 - ", data[data.Target==1].shape[0])

print("Target 2 - ", data[data.Target==2].shape[0])

print("Target 3 - ", data[data.Target==3].shape[0])

print("Target 4 - ", data[data.Target==4].shape[0])
targ_1 = data[data.Target==1].sample(frac=0.8)

targ_2 = data[data.Target==2].sample(frac=0.8)

targ_3 = data[data.Target==3].sample(frac=0.8)

targ_4 = data[data.Target==4].sample(frac=0.8)
train = pd.DataFrame

train  = pd.concat([targ_1, targ_2, targ_3, targ_4])

train = train.sample(frac=1) # Shuffle data
test = data.loc[~data.index.isin(train.index)]
X_train = train.drop(['Target'], axis =1)

Y_train = train['Target']

X_test = test.drop(['Target'], axis =1)

Y_test = test['Target']
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(70)

knn.fit(X_train, Y_train)
print('Train score for K-70 :', knn.score(X_train, Y_train)*100)

print('Train score for K-70 :',knn.score(X_test, Y_test)*100)
import pickle

knnPickle = open('knnpickle_file', 'wb')

pickle.dump(knn, knnPickle)
submission_x = pd.read_csv('../input/costa-rican-household-poverty-prediction/test.csv')
drop_columns(submission_x)

replce_na(submission_x)

get_cat_cols(submission_x)

encode_data(submission_x)
loaded_model = pickle.load(open('knnpickle_file', 'rb'))

result = loaded_model.predict(submission_x)
sample_sub = pd.read_csv('../input/costa-rican-household-poverty-prediction/sample_submission.csv')
sample_sub.Target = result
sample_sub.head(5)
sub = pd.DataFrame(columns=['Id','Target'])

sub.Id = sample_sub.Id

sub.Target = sample_sub.Target
sub.to_csv('submission', encoding='utf-8', index=False)