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
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#linear algebra

import numpy as np



#dataframe

import pandas as pd



#data visualization

import matplotlib.pyplot as plt

import seaborn as sns



#regex

import re

import sklearn

#machine learning

from sklearn.preprocessing import LabelEncoder,MinMaxScaler

from sklearn.model_selection import train_test_split,GridSearchCV, RandomizedSearchCV

from sklearn.metrics import classification_report,f1_score,accuracy_score

from xgboost import XGBClassifier



from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB



from vecstack import stacking



#neural network

import tensorflow as tf

from tensorflow import keras



import missingno



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rnd

from scipy import stats

import copy 

import warnings

warnings.filterwarnings('ignore')



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine-learning

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV

from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from sklearn.feature_selection import RFE, RFECV

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier, GradientBoostingClassifier

#from xgboost import XGBClassifier

#from imblearn.over_sampling import SMOTE

#from sklearn.ensemble import BaggingClassifier
#import

train = pd.read_csv('../input/prsits/train.csv')

test = pd.read_csv('../input/prsits/test.csv')
test.head(20)
#ganti nama duls

train.rename({'default.payment.next.month' : 'skip'}, axis=1, inplace=True)
# create bins for age

def age_group_fun(age):

    a = ''

    if age <= 0:

        a = 0

    else:

        a = 1

    return a

        

## Applying "age_group_fun" function to the "Age" column.

train['PAY_0'] = train['PAY_0'].map(age_group_fun)

test['PAY_0'] = test['PAY_0'].map(age_group_fun)

train['PAY_2'] = train['PAY_2'].map(age_group_fun)

test['PAY_2'] = test['PAY_2'].map(age_group_fun)

train['PAY_3'] = train['PAY_3'].map(age_group_fun)

test['PAY_3'] = test['PAY_3'].map(age_group_fun)

train['PAY_4'] = train['PAY_4'].map(age_group_fun)

test['PAY_4'] = test['PAY_4'].map(age_group_fun)

train['PAY_5'] = train['PAY_5'].map(age_group_fun)

test['PAY_5'] = test['PAY_5'].map(age_group_fun)

train['PAY_6'] = train['PAY_6'].map(age_group_fun)

test['PAY_6'] = test['PAY_6'].map(age_group_fun)
def sum_frame_by_column(frame, new_col_name, list_of_cols_to_sum):

    frame[new_col_name] = frame[list_of_cols_to_sum].astype(float).sum(axis=1)

    return(frame)
#sum_delay #sum_bill #sum_pay

sum_frame_by_column(train, 'sum_pay', ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5', 'PAY_AMT6'])

sum_frame_by_column(train, 'sum_delay', ['PAY_0', 'PAY_2', 'PAY_3','PAY_4', 'PAY_5','PAY_6'])

sum_frame_by_column(train, 'sum_bill', ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5', 'BILL_AMT6'])

sum_frame_by_column(test, 'sum_pay', ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5', 'PAY_AMT6'])

sum_frame_by_column(test, 'sum_delay', ['PAY_0', 'PAY_2', 'PAY_3','PAY_4', 'PAY_5','PAY_6'])

sum_frame_by_column(test, 'sum_bill', ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5', 'BILL_AMT6'])
train = train.drop(["BILL_AMT6","BILL_AMT5","BILL_AMT4","BILL_AMT3","BILL_AMT2","BILL_AMT1","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6","ID"],axis=1)

test = test.drop(["BILL_AMT6","BILL_AMT5","BILL_AMT4","BILL_AMT3","BILL_AMT2","BILL_AMT1","PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"],axis=1)
all_dummies = pd.get_dummies(test, drop_first = True)

all_dummies.head()
df = pd.concat([train[train.columns[:-1]], test],sort = False)

ed_bin_df = pd.get_dummies(df["EDUCATION"], prefix="ed")

mr_bin_df = pd.get_dummies(df["MARRIAGE"], prefix="mr")

gender_bin_df = pd.get_dummies(df["SEX"], prefix="g")



train = pd.concat([train, gender_bin_df.iloc[:len(train)], ed_bin_df.iloc[:len(train)],mr_bin_df.iloc[:len(train)]], axis = 1)

train.drop(["SEX"],axis=1,inplace=True)

train.drop(["EDUCATION"],axis=1,inplace=True)

train.drop(["MARRIAGE"],axis=1,inplace=True)



test = pd.concat([test, gender_bin_df.iloc[:len(test)], ed_bin_df.iloc[:len(test)],mr_bin_df.iloc[:len(test)]], axis = 1)

test.drop(["SEX"],axis=1,inplace=True)

test.drop(["EDUCATION"],axis=1,inplace=True)

test.drop(["MARRIAGE"],axis=1,inplace=True)
train
pd.DataFrame(abs(train.corr()['skip']).sort_values(ascending = False))
mask = np.zeros_like(train.corr(), dtype=np.bool)

## in order to reverse the bar replace "RdBu" with "RdBu_r"

plt.subplots(figsize = (15,12))

sns.heatmap(train.corr(), annot=True,mask = False,cmap = 'OrRd', linewidths=.7, linecolor='black',fmt='.2g',center = 0,square=True)



plt.title("Correlations", y = 1.03,fontsize = 20, fontweight = 'bold', pad = 40);
feature_to_scale = ["LIMIT_BAL", "AGE","sum_pay","sum_bill"]



minmax_scaler = MinMaxScaler()

df_temp = pd.concat([train.drop(["skip"],axis=1), test],sort=False)

minmax_scaler.fit(df_temp[feature_to_scale],)



train[feature_to_scale] = minmax_scaler.transform(train[feature_to_scale])

test[feature_to_scale] = minmax_scaler.transform(test[feature_to_scale])
train
X_train = train.drop("skip",axis=1)

Y_train = train["skip"]

X_test  = test.drop("ID",axis=1).copy()
logreg = LogisticRegression()



logreg.fit(X_train, Y_train) #model cari persamaan



Y_pred = logreg.predict(X_test) #prediksi



logreg.score(X_train, Y_train)