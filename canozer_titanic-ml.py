# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt  # Matlab-style plotting

import seaborn as sns

color = sns.color_palette()

sns.set_style('darkgrid')

import warnings

def ignore_warn(*args, **kwargs):

    pass

warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)





from scipy import stats

from scipy.stats import norm, skew #for some statistics

import missingno as msno

from sklearn.preprocessing import LabelEncoder





pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")
#check the numbers of samples and features

print("The train data size before dropping Id feature is : {} ".format(train.shape))

print("The test data size before dropping Id feature is : {} ".format(test.shape))
#copy trainand test ids

train_id = train["PassengerId"]

test_id =test["PassengerId"]
# drop PassengerId columns

train.drop(columns=["PassengerId"], inplace = True)

test.drop(columns=["PassengerId"], inplace = True)
print("The train data size after dropping Id feature is : {} ".format(train.shape))

print("The test data size after dropping Id feature is : {} ".format(test.shape))
msno.matrix(train)
plt.figure()

sns.boxplot("Fare",data=train)

plt.figure()

sns.boxplot("Age",data=train)
train = train[train["Fare"]< 500]

train = train[train["Age"] < 80]
n_train = train.shape[0]

n_test = test.shape[0]

y_train = train.Survived.values

all_data = pd.concat([train,test]).reset_index(drop = True)

all_data.drop(columns = ["Survived"],inplace = True)

print("all data shape is", all_data.shape)
all_data_na = pd.DataFrame(all_data.isna().sum()/len(all_data)*100, columns =["Missing Ratio"])

all_data_na.sort_values(by=["Missing Ratio"], ascending = False, inplace = True)

all_data_na
plt.figure(figsize = (10,8))

sns.barplot(x=all_data_na.index,y=all_data_na["Missing Ratio"])
# Filling NaN values

all_data["Cabin"] = all_data["Cabin"].fillna("None")

all_data["Age"] = all_data["Age"].fillna(all_data["Age"].median())

all_data["Fare"] = all_data["Fare"].fillna(all_data["Fare"].median())
col = all_data.columns.to_list()



for i in col:

    print(all_data[i].value_counts())
def format_name(df):

    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])

    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])

    return df   

def drop_features(df):

    df.drop(columns =["Ticket","Embarked","Name"],inplace = True)

    return df

format_name(all_data)

drop_features(all_data)
# Splitting numerical and categorical features

object_features = all_data.select_dtypes(include = "object")

numerical_features = all_data.select_dtypes(exclude = "object")
le = LabelEncoder()

ob_col=object_features.columns.to_list()

for i in ob_col:

    le.fit(object_features[i])

    object_features[i] = le.transform(object_features[i])
from sklearn.preprocessing import MinMaxScaler

num_cols = ["Age","Fare"]

Nm = MinMaxScaler()

for i in num_cols:

    Nm.fit(numerical_features[[i]].values)

    numerical_features[i] = Nm.transform(numerical_features[[i]])
# Combine numerical and categorical features

all_data = pd.concat([object_features,numerical_features],axis =1)



# split all_data into train and test sets

train = all_data[:n_train]

test = all_data[n_train:]

print("Train set:", train.shape)

print("Test set:", test.shape)
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from sklearn.model_selection import GridSearchCV 

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import LinearSVC

from sklearn.linear_model import RidgeClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import GradientBoostingClassifier

import xgboost as xgb

import lightgbm as lgb
n_folds = 5 

kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)



def model_eva(model):

    score = cross_val_score(model, train.values, y_train, cv = kf)

    return score
DTC = DecisionTreeClassifier()

KNC = KNeighborsClassifier()

LSVC = LinearSVC()

RC = RidgeClassifier()

SGDC = SGDClassifier()

xg = xgb.XGBRFClassifier()

lg = lgb.LGBMClassifier()

GBC = GradientBoostingClassifier()
score = model_eva(DTC)

print("Decision Tree Classifier score: {:.4f} and Std.: Dev:{:.4f}".format(score.mean(), score.std()))

score = model_eva(KNC)

print("KNeighborsClassifier score: {:.4f} and Sdt Dev.: {:.4f}".format(score.mean(), score.std()))

score = model_eva(LSVC)

print("Linear Support Vector Classifier score: {:.4f} and Sdt Dev.: {:.4f}".format(score.mean(),score.std()))

score = model_eva(RC)

print("RidgeClassifier score: {:.4f} and Sdt Dev.: {:.4f}".format(score.mean(),score.std()))

score = model_eva(SGDC)

print("SGDC score: {:.4f} and Sdt Dev.: {:.4f}".format(score.mean(),score.std()))

score = model_eva(xg)

print("XGB score: {:.4f} and Sdt Dev.: {:.4f}".format(score.mean(),score.std()))

score = model_eva(lg)

print("LGB score: {:.4f} and Sdt Dev.: {:.4f}".format(score.mean(),score.std()))

score = model_eva(GBC)

print("GBC score: {:.4f} and Sdt Dev.: {:.4f}".format(score.mean(),score.std()))
xg.fit(train, y_train)

xgb_train_pred = xg.predict(train)

xgb_pred = xg.predict(test)



lg.fit(train, y_train)

lgb_train_pred = lg.predict(train)

lgb_pred = lg.predict(test)



GBC.fit(train, y_train)

GBC_train_pred = GBC.predict(train)

GBC_pred = GBC.predict(test)

ensemble = xgb_train_pred*0.3+GBC_train_pred*0.3+lgb_train_pred*0.4

ensemble = ensemble.round()

accuracy_score(y_train,ensemble)
ensemble_pred = xgb_pred*0.3+GBC_pred*0.3+lgb_pred*0.4

ensemble_pred = np.rint(ensemble_pred)
sub = pd.DataFrame()

sub["PassengerId"] = test_id

sub["Survived"] = ensemble_pred.astype("int64")

sub.to_csv("Submission.csv",index = False)