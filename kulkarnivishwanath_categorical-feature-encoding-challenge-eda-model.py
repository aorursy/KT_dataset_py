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

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 500)



from sklearn.model_selection import train_test_split,RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import StratifiedKFold

import xgboost as xgb

import lightgbm as lgb

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_score,cross_val_predict
train = pd.read_csv("../input/cat-in-the-dat/train.csv")

test = pd.read_csv("../input/cat-in-the-dat/test.csv")
train.shape, test.shape
train.dtypes.value_counts()
test.dtypes.value_counts()
train.drop("id",axis=1,inplace=True)

Submission = test[['id']]

test.drop('id',axis=1,inplace=True)
train.head()
for col in train.columns:

    if train[col].nunique() == 2:

        print ("Value Counts of {} Binary Variable:\n".format(col),train[col].value_counts())

        print ("---------------------------------------")
cols = ['bin_0','bin_1','bin_2','bin_3','bin_4']

for ind,col in enumerate(train[cols]):

    plt.figure(ind)

    sns.countplot(x=col,data=train,hue='target')
for col in ['bin_0','bin_1','bin_2','bin_3','bin_4']:

    print ("Value Count of {} Variable grouped by the target variable:\n".format(col),train.groupby(col)['target'].value_counts())
nominal_cols = [col for col in train.columns if col.startswith("nom")]

train[nominal_cols].head()
for col in nominal_cols:

    print ("Unique Values in {} Nominal Variable:".format(col),train[col].nunique())

    print ("-------------------------------------------------------")
cols = ['nom_0','nom_1','nom_2','nom_3','nom_4']

for ind,col in enumerate(train[cols]):

    plt.figure(ind)

    sns.countplot(x=col,data=train)
train.groupby('nom_0')['target'].value_counts().unstack().plot(kind='bar')

plt.title("Distribution of Target variable by nom_0 variable")

plt.xlabel("nom_0 variable")

plt.ylabel("Count")
train.groupby('nom_1')['target'].value_counts().unstack().plot(kind='bar')

plt.title("Distribution of Target variable by nom_1 variable")

plt.xlabel("nom_1 variable")

plt.ylabel("Count")
train.groupby('nom_2')['target'].value_counts().unstack().plot(kind='bar')

plt.title("Distribution of Target variable by nom_2 variable")

plt.xlabel("nom_2 variable")

plt.ylabel("Count")
train.groupby('nom_3')['target'].value_counts().unstack().plot(kind='bar')

plt.title("Distribution of Target variable by nom_3 variable")

plt.xlabel("nom_3 variable")

plt.ylabel("Count")
train.groupby('nom_4')['target'].value_counts().unstack().plot(kind='bar')

plt.title("Distribution of Target variable by nom_4 variable")

plt.xlabel("nom_4 variable")

plt.ylabel("Count")
high_card_nominal = [col for col in train.columns if((col.startswith("nom"))&(train[col].nunique()>10))]

train[high_card_nominal].head()
for col in high_card_nominal:

    print ("Cardinality of {} nominal variable is:".format(col),train[col].nunique())
for col in high_card_nominal:

    print ("Value Count of Top 5 values in the {} nominal variable:".format(col),train[col].value_counts().sort_values(ascending=False).head())

    print ("---------------------------------------------------------------------------------")
ordinal_cols = [col for col in train.columns if col.startswith("ord")]

train[ordinal_cols].head()
for col in ordinal_cols:

    print ("Number of Unique values in {} Ordinal Variable are:".format(col),train[col].nunique())
for col in ordinal_cols:

    print ("Value Count of Top 5 values in the {} ordinal variable:\n".format(col),train[col].value_counts().sort_values(ascending=False).head())

    print ("---------------------------------------------------------------------------------")
train.groupby('ord_0')['target'].value_counts().unstack().plot(kind='bar')

plt.title("Distribution of Target Variable by ord_0 variable")

plt.xlabel("ord_0 variable")

plt.ylabel("Counts")
train.groupby('ord_1')['target'].value_counts().sort_values().unstack().plot(kind='bar')

plt.title("Distribution of Target Variable by ord_1 variable")

plt.xlabel("ord_1 variable")

plt.ylabel("Counts")
train.groupby('ord_2')['target'].value_counts().sort_values().unstack().plot(kind='bar')

plt.title("Distribution of Target Variable by ord_2 variable")

plt.xlabel("ord_2 variable")

plt.ylabel("Counts")
train.groupby('ord_3')['target'].value_counts().sort_values().unstack().plot(kind='bar')

plt.title("Distribution of Target Variable by ord_3 variable")

plt.xlabel("ord_3 variable")

plt.ylabel("Counts")
train.groupby('ord_4')['target'].value_counts().unstack().plot(kind='bar',figsize=(10,6))

plt.title("Distribution of Target Variable by ord_4 variable")

plt.xlabel("ord_4 variable")

plt.ylabel("Counts")
train[['day','month']].head()
for col in ['day','month']:

    print ("Unique Values in {} variable are:\n".format(col),train[col].nunique())
train.groupby('day')['target'].value_counts().sort_values().unstack().plot(kind='bar')

plt.title("Distribution of Target Variable by Day")

plt.xlabel("Day of the Week")

plt.ylabel("Count")
train.groupby('month')['target'].value_counts().sort_values().unstack().plot(kind='bar')

plt.title("Distribution of Target Variable by Month")

plt.xlabel("Month")

plt.ylabel("Count")
# Converting bin_3 and bin_4 variables in the form of 0's and 1's

mapping = {"T":1,"F":0,"Y":1,"N":0}

train['bin_4'] = train['bin_4'].map(mapping)

train['bin_3'] = train['bin_3'].map(mapping)



# Converting ordinal columns ord_0,ord_1,ord_2,ord_3,ord_4 to category data type with the assumed ordering

train['ord_0'] = train['ord_0'].astype('category')

train['ord_0'] = train['ord_0'].cat.set_categories([1,2,3],ordered=True)

train['ord_0'] = train['ord_0'].cat.codes



train['ord_1'] = train['ord_1'].astype('category')

train['ord_1'] = train['ord_1'].cat.set_categories(["Novice","Contributor","Expert","Master","Grandmaster"],ordered=True)

train['ord_1'] = train['ord_1'].cat.codes





train['ord_2'] = train['ord_2'].astype('category')

train['ord_2'] = train['ord_2'].cat.set_categories(["Freezing","Cold","Warm","Hot","Boiling Hot","Lava Hot"],ordered=True)

train['ord_2'] = train['ord_2'].cat.codes



train['ord_3'] = train['ord_3'].astype('category')

train['ord_3'] = train['ord_3'].cat.set_categories(["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o"],ordered=True)

train['ord_3'] = train['ord_3'].cat.codes



train['ord_4'] = train['ord_4'].astype('category')

train['ord_4'] = train['ord_4'].cat.set_categories(["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],ordered=True)

train['ord_4'] = train['ord_4'].cat.codes



train['ord_5'] = train['ord_5'].astype('category')

train['ord_5'] = train['ord_5'].cat.codes



# Converting day and month variables to category data type

train['day'] = train['day'].astype('category')

train['day'] = train['day'].cat.codes

train['month'] = train['month'].astype('category')

train['month'] = train['month'].cat.codes



# Dummy encoding

nom_0_dummy = pd.get_dummies(train['nom_0'],prefix="nom_0",)

train = pd.concat([train,nom_0_dummy],axis=1)

train.drop("nom_0",axis=1,inplace=True)



nom_1_dummy = pd.get_dummies(train['nom_1'],prefix="nom_1")

train = pd.concat([train,nom_1_dummy],axis=1)

train.drop("nom_1",axis=1,inplace=True)



nom_2_dummy = pd.get_dummies(train['nom_2'],prefix="nom_2")

train = pd.concat([train,nom_2_dummy],axis=1)

train.drop("nom_2",axis=1,inplace=True)



nom_3_dummy = pd.get_dummies(train['nom_3'],prefix="nom_3")

train = pd.concat([train,nom_3_dummy],axis=1)

train.drop("nom_3",axis=1,inplace=True)



nom_4_dummy = pd.get_dummies(train['nom_4'],prefix="nom_4")

train = pd.concat([train,nom_4_dummy],axis=1)

train.drop("nom_4",axis=1,inplace=True)



day_dummy = pd.get_dummies(train['day'],prefix="day")

train = pd.concat([train,day_dummy],axis=1)

train.drop("day",axis=1,inplace=True)



month_dummy = pd.get_dummies(train['month'],prefix="month")

train = pd.concat([train,month_dummy],axis=1)

train.drop("month",axis=1,inplace=True)





nom_5_freq_encoding = train['nom_5'].value_counts().to_dict()

train['nom_5_freq_encoding'] = train['nom_5'].map(nom_5_freq_encoding)



nom_6_freq_encoding = train['nom_6'].value_counts().to_dict()

train['nom_6_freq_encoding'] = train['nom_6'].map(nom_6_freq_encoding)



nom_7_freq_encoding = train['nom_7'].value_counts().to_dict()

train['nom_7_freq_encoding'] = train['nom_7'].map(nom_7_freq_encoding)



nom_8_freq_encoding = train['nom_8'].value_counts().to_dict()

train['nom_8_freq_encoding'] = train['nom_8'].map(nom_8_freq_encoding)



nom_9_freq_encoding = train['nom_9'].value_counts().to_dict()

train['nom_9_freq_encoding'] = train['nom_9'].map(nom_9_freq_encoding)





nom_5_target_encoding = np.round(train.groupby('nom_5')['target'].mean(),decimals=2).to_dict()

train['nom_5_target_encoding'] = train['nom_5'].map(nom_5_target_encoding)



nom_6_target_encoding = np.round(train.groupby('nom_6')['target'].mean(),decimals=2).to_dict()

train['nom_6_target_encoding'] = train['nom_6'].map(nom_6_target_encoding)



nom_7_target_encoding = np.round(train.groupby('nom_7')['target'].mean(),decimals=2).to_dict()

train['nom_7_target_encoding'] = train['nom_7'].map(nom_7_target_encoding)



nom_8_target_encoding = np.round(train.groupby('nom_8')['target'].mean(),decimals=2).to_dict()

train['nom_8_target_encoding'] = train['nom_8'].map(nom_8_target_encoding)



nom_9_target_encoding = np.round(train.groupby('nom_9')['target'].mean(),decimals=2).to_dict()

train['nom_9_target_encoding'] = train['nom_9'].map(nom_9_target_encoding)
# Converting bin_3 and bin_4 variables in the form of 0's and 1's

mapping = {"T":1,"F":0,"Y":1,"N":0}

test['bin_4'] = test['bin_4'].map(mapping)

test['bin_3'] = test['bin_3'].map(mapping)



# Converting ordinal columns ord_0,ord_1,ord_2,ord_3,ord_4 to category data type with the assumed ordering

test['ord_0'] = test['ord_0'].astype('category')

test['ord_0'] = test['ord_0'].cat.set_categories([1,2,3],ordered=True)

test['ord_0'] = test['ord_0'].cat.codes



test['ord_1'] = test['ord_1'].astype('category')

test['ord_1'] = test['ord_1'].cat.set_categories(["Novice","Contributor","Expert","Master","Grandmaster"],ordered=True)

test['ord_1'] = test['ord_1'].cat.codes





test['ord_2'] = test['ord_2'].astype('category')

test['ord_2'] = test['ord_2'].cat.set_categories(["Freezing","Cold","Warm","Hot","Boiling Hot","Lava Hot"],ordered=True)

test['ord_2'] = test['ord_2'].cat.codes



test['ord_3'] = test['ord_3'].astype('category')

test['ord_3'] = test['ord_3'].cat.set_categories(["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o"],ordered=True)

test['ord_3'] = test['ord_3'].cat.codes



test['ord_4'] = test['ord_4'].astype('category')

test['ord_4'] = test['ord_4'].cat.set_categories(["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"],ordered=True)

test['ord_4'] = test['ord_4'].cat.codes



test['ord_5'] = test['ord_5'].astype('category')

test['ord_5'] = test['ord_5'].cat.codes



# Converting day and month variables to category data type

test['day'] = test['day'].astype('category')

test['day'] = test['day'].cat.codes

test['month'] = test['month'].astype('category')

test['month'] = test['month'].cat.codes



# Dummy encoding

nom_0_dummy = pd.get_dummies(test['nom_0'],prefix="nom_0",)

test = pd.concat([test,nom_0_dummy],axis=1)

test.drop("nom_0",axis=1,inplace=True)



nom_1_dummy = pd.get_dummies(test['nom_1'],prefix="nom_1")

test = pd.concat([test,nom_1_dummy],axis=1)

test.drop("nom_1",axis=1,inplace=True)



nom_2_dummy = pd.get_dummies(test['nom_2'],prefix="nom_2")

test = pd.concat([test,nom_2_dummy],axis=1)

test.drop("nom_2",axis=1,inplace=True)



nom_3_dummy = pd.get_dummies(test['nom_3'],prefix="nom_3")

test = pd.concat([test,nom_3_dummy],axis=1)

test.drop("nom_3",axis=1,inplace=True)



nom_4_dummy = pd.get_dummies(test['nom_4'],prefix="nom_4")

test = pd.concat([test,nom_4_dummy],axis=1)

test.drop("nom_4",axis=1,inplace=True)



day_dummy = pd.get_dummies(test['day'],prefix="day")

test = pd.concat([test,day_dummy],axis=1)

test.drop("day",axis=1,inplace=True)



month_dummy = pd.get_dummies(test['month'],prefix="month")

test = pd.concat([test,month_dummy],axis=1)

test.drop("month",axis=1,inplace=True)





nom_5_freq_encoding = test['nom_5'].value_counts().to_dict()

test['nom_5_freq_encoding'] = test['nom_5'].map(nom_5_freq_encoding)



nom_6_freq_encoding = test['nom_6'].value_counts().to_dict()

test['nom_6_freq_encoding'] = test['nom_6'].map(nom_6_freq_encoding)



nom_7_freq_encoding = test['nom_7'].value_counts().to_dict()

test['nom_7_freq_encoding'] = test['nom_7'].map(nom_7_freq_encoding)



nom_8_freq_encoding = test['nom_8'].value_counts().to_dict()

test['nom_8_freq_encoding'] = test['nom_8'].map(nom_8_freq_encoding)



nom_9_freq_encoding = test['nom_9'].value_counts().to_dict()

test['nom_9_freq_encoding'] = test['nom_9'].map(nom_9_freq_encoding)





test['nom_5_target_encoding'] = test['nom_5'].map(nom_5_target_encoding)

test['nom_6_target_encoding'] = test['nom_6'].map(nom_6_target_encoding)

test['nom_7_target_encoding'] = test['nom_7'].map(nom_7_target_encoding)

test['nom_8_target_encoding'] = test['nom_8'].map(nom_8_target_encoding)

test['nom_9_target_encoding'] = test['nom_9'].map(nom_9_target_encoding)
train.drop(['nom_5','nom_6','nom_7','nom_8','nom_9'],axis=1,inplace=True)

test.drop(['nom_5','nom_6','nom_7','nom_8','nom_9'],axis=1,inplace=True)
train.head()
test.head()
X = train[[col for col in train.columns if col!='target']]

y = train['target']
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

for train_index,test_index in kf.split(X,y):

    X_Train,X_Valid = X.loc[train_index],X.loc[test_index]

    y_Train,y_Valid = y.loc[train_index],y.loc[test_index]

print (X_Train.shape)

print (y_Train.shape)

print (X_Valid.shape)

print (y_Valid.shape)
clf_1 = lgb.LGBMClassifier(boosting_type='goss',objective='binary',random_state=42,n_jobs=-1,verbose=1,class_weight='balanced')

params = {"max_depth":[3,4,5,6,7,-1],

          "learning_rate":[0.01,0.05,0.1,0.3],

          "subsample":[0.5,0.6,0.7,0.8,0.9],

          "colsample_bytree":[0.5,0.6,0.7,0.8,0.9],

          "reg_alpha":[0.5,1,2,5,10],

          "reg_lambda":[0.5,1,2,5,10],

          "num_leaves":[7,15,31,63,127],

          "n_estimators":list(range(50,500,50)),

          "min_data_in_leaf":[1,3,5,10,15,25]}

random_search_1 = RandomizedSearchCV(estimator=clf_1,param_distributions=params,cv=10,scoring='roc_auc')

random_search_1.fit(X_Train,y_Train)
random_search_1.best_estimator_,random_search_1.best_score_,random_search_1.best_params_
ser = pd.Series(random_search_1.best_estimator_.feature_importances_,X_Train.columns).sort_values()

ser.plot(kind='bar',figsize=(10,6))
lst = list(ser[ser>0].index)

X_Train = X_Train[lst]

test = test[lst]
clf_1 = lgb.LGBMClassifier(boosting_type='goss',objective='binary',random_state=42,n_jobs=-1,verbose=1,class_weight='balanced')

params = {"max_depth":[3,4,5,6,7,-1],

          "learning_rate":[0.01,0.05,0.1,0.3],

          "subsample":[0.5,0.6,0.7,0.8,0.9],

          "colsample_bytree":[0.5,0.6,0.7,0.8,0.9],

          "reg_alpha":[0.5,1,2,5,10],

          "reg_lambda":[0.5,1,2,5,10],

          "num_leaves":[7,15,31,63,127],

          "n_estimators":list(range(50,500,50)),

          "min_data_in_leaf":[1,3,5,10,15,25]}

random_search_1 = RandomizedSearchCV(estimator=clf_1,param_distributions=params,cv=10,scoring='roc_auc')

random_search_1.fit(X_Train,y_Train)
random_search_1.best_estimator_,random_search_1.best_score_,random_search_1.best_params_
"""clf_2 = xgb.XGBClassifier(random_state=42,n_jobs=-1,verbosity=1)

params = {"max_depth":[3,4,5,6,7,8,9],

          "n_estimators":list(range(50,500,50)),

          "learning_rate":[0.01,0.05,0.1,0.3],

          "subsample":[0.5,0.6,0.7,0.8,0.9],

          "colsample_bytree":[0.5,0.6,0.7,0.8,0.9],

          "reg_alpha":[0.5,1,2,5,10],

          "reg_lambda":[0.5,1,2,5,10]}

random_search_2 = RandomizedSearchCV(estimator=clf_2,param_distributions=params,cv=10,scoring='roc_auc')

random_search_2.fit(X_Train,y_Train)"""
Submission['target']=random_search_1.predict_proba(test)[:,1]

Submission.to_csv("Latest.csv",index=None)