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

pd.set_option("display.max_columns",50)



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from sklearn.preprocessing import LabelEncoder,OneHotEncoder,OrdinalEncoder,StandardScaler

from sklearn.model_selection import train_test_split,GridSearchCV,KFold,RandomizedSearchCV,StratifiedKFold

import lightgbm as lgb

import xgboost as xgb

from sklearn.metrics import roc_auc_score

import pandas_profiling
dtypes = {"day":"float32","month":"float32","target":"uint8","bin_0":"float32","bin_1":"float32","bin_2":"float32","ord_0":"float32"}

train = pd.read_csv("../input/cat-in-the-dat-ii/train.csv",dtype=dtypes)

test = pd.read_csv("../input/cat-in-the-dat-ii/test.csv",dtype=dtypes)
train.info(memory_usage='deep')
test.info(memory_usage='deep')
train.drop("id",axis=1,inplace=True)

Submission = test[['id']]

test.drop('id',axis=1,inplace=True)
train.head()
test.head()
sns.countplot(train['target'],)

plt.title("Distribution of Dependent Variable")

plt.xlabel("Target")

plt.ylabel("Count")
train['target'].value_counts(normalize=True)
cols = [col for col in train.columns if col!='target']

bin_cols = ['bin_0','bin_1','bin_2','bin_3','bin_4']

ord_cols = ['ord_0','ord_1','ord_2','ord_3','ord_4','ord_5']

nom_cols = ['nom_0','nom_1','nom_2','nom_3','nom_4','nom_5','nom_6','nom_7','nom_8','nom_9']

print ("Dependent Variables are:{}".format(cols))
for col in cols:

    print ("Unique Values in {} variable in Train data are:{}".format(col,train[col].nunique()))

    print ("Unique Values in {} variable in Test data are:{}".format(col,test[col].nunique()))

    print ("--------------------------------------------------------------------------------")
for col in cols:

    print ("Percentage of Missing Values in {} variable in Train data are:{}".format(col,train[col].isna().sum()/len(train)))

    print ("Percentage of Missing Values in {} variable in Test data are:{}".format(col,test[col].isna().sum()/len(test)))

    print ("--------------------------------------------------------------------------------")
for col in bin_cols:

    train[col].fillna(train[col].value_counts().index[0],inplace=True)

    test[col].fillna(test[col].value_counts().index[0],inplace=True)
# Converting bin_3 and bin_4 variables in the form of 0's and 1's

mapping = {"T":1,"F":0,"Y":1,"N":0}

train['bin_4'] = train['bin_4'].map(mapping)

train['bin_3'] = train['bin_3'].map(mapping)



test['bin_4'] = test['bin_4'].map(mapping)

test['bin_3'] = test['bin_3'].map(mapping)



# converting the float values to int

for col in ['bin_0','bin_1','bin_2']:

    train[col] = train[col].astype('int')

    test[col] = test[col].astype('int')
for ind,col in enumerate(train[bin_cols]):

    plt.figure(ind)

    sns.countplot(x=col,data=train,hue='target')
for col in ['bin_0','bin_1','bin_2','bin_3','bin_4']:

    print ("Value Count of {} Variable grouped by the target variable:\n".format(col),train.groupby(col)['target'].value_counts(normalize=True))
for col in ord_cols:

    train[col].fillna(train[col].value_counts().index[0],inplace=True)

    test[col].fillna(test[col].value_counts().index[0],inplace=True)
for ind,col in enumerate(train[ord_cols]):

    plt.figure(figsize=(14,6))

    plt.figure(ind)

    sns.countplot(x=col,data=train,order=train[col].value_counts().index.values,orient='h')

    plt.xticks(rotation=90)
le = LabelEncoder()

for df in [train,test]:

    df['ord_5'] = le.fit_transform(df['ord_5'])



# Converting ordinal columns ord_0,ord_1,ord_2,ord_3,ord_4 to category data type with the assumed ordering

train['ord_0'] = train['ord_0'].astype('category')

train['ord_0'] = train['ord_0'].cat.set_categories([1.0,2.0,3.0],ordered=True)

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
# Filling missing values in the nominal columns with the mode.

for col in nom_cols:

    train[col].fillna(train[col].value_counts().index[0],inplace=True)

    test[col].fillna(test[col].value_counts().index[0],inplace=True)
cols = ['nom_0','nom_1','nom_2','nom_3','nom_4']

for ind,col in enumerate(train[cols]):

    plt.figure(ind)

    sns.countplot(x=col,data=train,order=train[col].value_counts().index.values,hue='target')
# Dummy variables of nominal variables with low cardinality



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
# Mean encoding the nominal variables that have hign cardinality

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





test['nom_5_target_encoding'] = test['nom_5'].map(nom_5_target_encoding)

test['nom_6_target_encoding'] = test['nom_6'].map(nom_6_target_encoding)

test['nom_7_target_encoding'] = test['nom_7'].map(nom_7_target_encoding)

test['nom_8_target_encoding'] = test['nom_8'].map(nom_8_target_encoding)

test['nom_9_target_encoding'] = test['nom_9'].map(nom_9_target_encoding)



test['nom_6_target_encoding'].fillna(test['nom_6_target_encoding'].mean(),inplace=True)



train.drop(['nom_5','nom_6','nom_7','nom_8','nom_9'],axis=1,inplace=True)

test.drop(['nom_5','nom_6','nom_7','nom_8','nom_9'],axis=1,inplace=True)
# Handling Cyclical Features such as day and month

for df in [train,test]:

    df['day'].fillna(df['day'].value_counts().index[0],inplace=True)

    df['month'].fillna(df['month'].value_counts().index[0],inplace=True)
# Sine and Cosine transformation of the cyclical features such as day and month

def date_cyc_enc(df, col, max_vals):

    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)

    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)

    return df



train = date_cyc_enc(train, 'day', 7)

test = date_cyc_enc(test, 'day', 7) 



train = date_cyc_enc(train, 'month', 12)

test = date_cyc_enc(test, 'month', 12)

train.drop(['day','month'],axis=1,inplace=True)

test.drop(['day','month'],axis=1,inplace=True)
cols_to_transform = ['ord_0','ord_1','ord_2','ord_3','ord_4','ord_5']

scaled_train = train.copy()

features_train = scaled_train[cols_to_transform]

scaler = StandardScaler().fit(features_train.values)

features_train = scaler.transform(features_train.values)

scaled_train[cols_to_transform] = features_train



scaled_test = test.copy()

features_test = scaled_test[cols_to_transform]

scaler = StandardScaler().fit(features_test.values)

features_test = scaler.transform(features_test.values)

scaled_test[cols_to_transform] = features_test
X = scaled_train[[col for col in scaled_train.columns if col!='target']]

y = scaled_train['target']
X_Train,X_Test,y_Train,y_Test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

print (X_Train.shape)

print (X_Test.shape)

print (y_Train.shape)

print (y_Test.shape)
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
Submission['target']=random_search_1.predict_proba(scaled_test)[:,1]

Submission.to_csv("Latest.csv",index=None)
Submission