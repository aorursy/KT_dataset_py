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

import scipy as scp

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

from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

from sklearn.ensemble import BaggingClassifier
import pandas as pd

test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")
#menampilkan sebagian data

train.sample(5)
test.sample(5)
#Mengitung nilai yang unik

for col in train.columns:

    print('Nilai unik pada feature', col, 'adalah')

    print(train[col].value_counts())

    print('\n')
#Melihat kategori data

print('Pada dataset flight terdapat beberapa feature, yaitu: ')

print('\n')

num_features = train.select_dtypes(['float64', 'int64']).columns.tolist()

cat_features = train.select_dtypes(['object']).columns.tolist()

print('{} numerical features:\n{} \n{} categorical features:\n{}'.format(len(num_features), num_features, len(cat_features), cat_features))

print('\n')

print('\n')

print('Sedangkan, pada dataset test terdapat beberapa feature, yaitu: ')

print('\n')

num_features = test.select_dtypes(['float64', 'int64']).columns.tolist()

cat_features = test.select_dtypes(['object']).columns.tolist()

print('{} numerical features:\n{} \n{} categorical features:\n{}'.format(len(num_features), num_features, len(cat_features), cat_features))
#Melihat data yang hilang

def missing_percentage(df):

    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""

    total = df.isnull().sum().sort_values(ascending = False)

    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)

    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])
missing_percentage(train)
missing_percentage(test)
#mengisi coloum country denganm modus yaitu PRT

train["country"]=train['country'].fillna("PRT")

test["country"]=test['country'].fillna("PRT")
#mengisi coloum children dengan modus yaitu 0

train["children"]=train['children'].fillna(0)
##pusing mengenai agent dan company. keputusan : dihapus

train.drop(["company"],axis=1,inplace=True)

train.drop(["agent"],axis=1,inplace=True)

test.drop(["company"],axis=1,inplace=True)

test.drop(["agent"],axis=1,inplace=True)

train.drop(["order_id"],axis=1,inplace=True)

train.drop(["reservation_status"],axis=1,inplace=True)

test.drop(["reservation_status"],axis=1,inplace=True)

train.drop(["reservation_status_date"],axis=1,inplace=True)

test.drop(["reservation_status_date"],axis=1,inplace=True)
missingno.matrix(train)
missingno.matrix(test)
train
pd.crosstab(train.hotel, train.is_canceled, normalize='index')
pd.crosstab(train.arrival_date_year, train.is_canceled, normalize='index')
pd.crosstab(train.arrival_date_month, train.is_canceled, normalize='index')
pd.crosstab(train.meal, train.is_canceled, normalize='index')
pd.crosstab(train.market_segment, train.is_canceled, normalize='index')
pd.crosstab(train.deposit_type, train.is_canceled, normalize='index')
pd.crosstab(train.customer_type, train.is_canceled, normalize='index')
pd.crosstab(train.distribution_channel, train.is_canceled, normalize='index')
pd.crosstab(train.required_car_parking_spaces, train.is_canceled, normalize='index')
pd.crosstab(train.reserved_room_type, train.is_canceled, normalize='index')
pd.crosstab(train.total_of_special_requests, train.is_canceled, normalize='index')
pd.crosstab(train.is_repeated_guest, train.is_canceled, normalize='index')
pd.crosstab(train.reserved_room_type, train.is_canceled, normalize='index')
pd.crosstab(train.assigned_room_type, train.is_canceled, normalize='index')
pd.crosstab(train.country, train.is_canceled, normalize='index')
pd.crosstab(train.previous_cancellations, train.is_canceled, normalize='index')
a = sns.FacetGrid( train, hue = 'is_canceled', aspect=4 )

a.map(sns.kdeplot, 'adr', shade= True )

a.set(xlim=(0 , train['adr'].max()))

a.add_legend()
a = sns.FacetGrid( train, hue = 'is_canceled', aspect=4 )

a.map(sns.kdeplot, 'arrival_date_week_number', shade= True )

a.set(xlim=(0 , train['arrival_date_week_number'].max()))

a.add_legend()
a = sns.FacetGrid( train, hue = 'is_canceled', aspect=4 )

a.map(sns.kdeplot, 'arrival_date_day_of_month', shade= True )

a.set(xlim=(0 , train['arrival_date_day_of_month'].max()))

a.add_legend()
a = sns.FacetGrid( train, hue = 'is_canceled', aspect=4 )

a.map(sns.kdeplot, 'stays_in_week_nights', shade= True )

a.set(xlim=(0 , train['stays_in_week_nights'].max()))

a.add_legend()
a = sns.FacetGrid( train, hue = 'is_canceled', aspect=4 )

a.map(sns.kdeplot, 'lead_time', shade= True )

a.set(xlim=(0 , train['lead_time'].max()))

a.add_legend()
pd.DataFrame(abs(train.corr()['is_canceled']).sort_values(ascending = False))
missing_percentage(train)
train
def categorical_to_numerical(dataset, feature):

    dictionary = {}

    for value in dataset[feature].unique():

        index = np.where(dataset[feature].unique() == value)

        dictionary[value] = index[0]

    dataset[feature] = dataset[feature].map(dictionary).astype(int)
dataset = [train, test]

cat_feature = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment', 'distribution_channel', 'reserved_room_type', 'assigned_room_type', 'deposit_type', 'customer_type']

for data in dataset:

    for feature in cat_feature:

        categorical_to_numerical(data, feature)
train 
pd.DataFrame(abs(train.corr()['is_canceled']).sort_values(ascending = False))
X_train = train.drop("is_canceled",axis=1)

Y_train = train["is_canceled"]

X_test  = test.drop("order_id",axis=1).copy()
logreg = LogisticRegression()



logreg.fit(X_train, Y_train) #model cari persamaan



Y_pred = logreg.predict(X_test) #prediksi



logreg.score(X_train, Y_train)
random_forest = RandomForestClassifier(n_estimators=10)

#random_forest = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=10, max_features='sqrt', min_samples_split=5)



random_forest.fit(X_train, Y_train)



Y_pred_1 = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)
Y_pred_1
submission = pd.DataFrame({

        "order_id": test["order_id"],

        "is_canceled": Y_pred_1

    })

submission.to_csv('submission.csv', index=False)