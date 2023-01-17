# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.cm as cm

import matplotlib.pyplot as plt

import seaborn as sns

figure = plt.figure()



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read data

train = pd.read_csv("/kaggle/input/insurance-churn-prediction-weekend-hackathon/Insurance_Churn_ParticipantsData/Train.csv")

test = pd.read_csv("/kaggle/input/insurance-churn-prediction-weekend-hackathon/Insurance_Churn_ParticipantsData/Test.csv")
train.head()
test.head()
train.describe()
train.dtypes
train.isnull().sum()
for col in train.columns:

    plot = plt.boxplot(train[col])

    print(f'plot of feature {col} is {plot}')

    plt.show()
train[train['feature_1']>24].index
train[train['feature_3']>15].index
train[train['feature_4']>16].index
train[train['feature_6']>20].index
train1 = train.drop([5445, 5606, 29608, 20042, 17893, 20894, 32159, 7705])
plt.figure(figsize=(16,8))

corr = train.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
X = train[[col for col in train.columns if not col == 'labels']]

X = X.set_index('feature_0')

X.shape
y = train['labels']

y.shape
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y,train_size=0.8, test_size=0.2,random_state = 0)
from xgboost import XGBClassifier

model_xgb  = XGBClassifier(n_estimators = 178,

                       eta = 0.17,

                       booster_pram = 'dart',

                       tree_method = 'hist',

                       scale_pos_weight= 5,

                       max_bin=215,

                       random_state = 0)
model_xgb.fit(train_X,

          train_y)
predict = model_xgb.predict(val_X)
from sklearn.metrics import f1_score

f1_score(val_y,predict)
from sklearn.metrics import confusion_matrix

print("Confusion matrix \n",confusion_matrix(val_y,predict))