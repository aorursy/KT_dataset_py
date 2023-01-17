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
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import accuracy_score,classification_report,roc_auc_score,roc_curve

from lightgbm import LGBMClassifier,plot_importance

from sklearn.preprocessing import StandardScaler
train = pd.read_csv('/kaggle/input/customer-churn-prediction-2020/train.csv')

test = pd.read_csv('/kaggle/input/customer-churn-prediction-2020/test.csv')

train.shape,test.shape
train.head(3)
train.state = pd.Categorical(train.state).codes

train.area_code = pd.Categorical(train.area_code).codes

train.international_plan = pd.Categorical(train.international_plan).codes

train.voice_mail_plan = pd.Categorical(train.voice_mail_plan).codes

train.churn = pd.Categorical(train.churn).codes

test.state = pd.Categorical(test.state).codes

test.area_code = pd.Categorical(test.area_code).codes

test.international_plan = pd.Categorical(test.international_plan).codes

test.voice_mail_plan = pd.Categorical(test.voice_mail_plan).codes
train.info()
sns.set_style('dark')

train.hist(bins=50,figsize=(20,20),color='navy')
X = train.drop('churn',axis=1)

y  = train.churn

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1)

from yellowbrick.classifier import roc_auc

roc_auc(LGBMClassifier(),X_train,y_train,X_test = X_test, y_test = y_test,

       classes=[0,1])
X_train = train.drop('churn',axis=1)

y_train = train.churn

X_test = test.drop('id',axis=1)
scaler = StandardScaler()

X_train.number_vmail_messages = scaler.fit_transform(X_train.number_vmail_messages.values.reshape(-1,1))

X_test.number_vmail_messages = scaler.fit_transform(X_test.number_vmail_messages.values.reshape(-1,1))
X_train.number_vmail_messages.hist(bins=100,color='navy')
light = LGBMClassifier(n_estimators=200,learning_rate=0.11,

                      min_child_samples=30,num_leaves=60)

light.fit(X_train,y_train)
pred = light.predict(X_test)
plt.rcParams["figure.figsize"] = (14, 7)

plot_importance(light,color='navy',)