import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import time



from sklearn.model_selection import RandomizedSearchCV,GridSearchCV,train_test_split

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler

from imblearn.over_sampling import SMOTE

from imblearn.combine import SMOTETomek

import xgboost as xgb

import lightgbm as lgb



import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 500)
%%time

train_transaction = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")

test_transaction = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")

train_identity = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")

test_identity = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")
print ("Training Transaction Dataset Shape: {}".format(train_transaction.shape))

print ("Testing Transaction Dataset Shape: {}".format(test_transaction.shape))

print ("Training Identity Dataset Shape: {}".format(train_identity.shape))

print ("Testing Identity Dataset Shape: {}".format(test_identity.shape))
print ("Unique Transaction ID's in Training Set",train_transaction['TransactionID'].nunique())

print ("Unique Transaction ID's in Training Identity Set",train_identity['TransactionID'].nunique())

print ("-------------------------------------------------------------")

print ("Unique Transaction ID's in Testing Set",test_transaction['TransactionID'].nunique())

print ("Unique Transaction ID's in Testing Identity Set",test_identity['TransactionID'].nunique())
%%time

train = pd.merge(left=train_transaction,right=train_identity,on='TransactionID',how='left')

test = pd.merge(left=test_transaction,right=test_identity,on='TransactionID',how='left')
del train_transaction, test_transaction, train_identity, test_identity
train.drop("TransactionID",axis=1,inplace=True)

Submission = test[['TransactionID']]

test.drop("TransactionID",axis=1,inplace=True)
print (train.shape)

print (test.shape)
missing_train = []

missing_test = []

for col in train.columns:

    if train[col].isna().sum() / len(train) > 0.90:

        missing_train.append(col)

for col in test.columns:

    if test[col].isna().sum() / len(test) > 0.90:

        missing_test.append(col)

print ("Following are the columns in Training Data with more than 90% missing values:",missing_train)

print ("Following are the columns in Test Data with more than 90% missing values:",missing_test)
common_elements = list(set(missing_train).intersection(missing_test))

common_elements
train.drop(common_elements,axis=1,inplace=True)

test.drop(common_elements,axis=1,inplace=True)
train['ProductCD'].nunique()

# There are 5 Levels in ProductCD variable. 
train['ProductCD'].value_counts(dropna=False)

# Product Code of W takes the maximum values and S takes the least. There are no missing values in the variable.
train.groupby('ProductCD')['isFraud'].value_counts(dropna=False).unstack()
train.groupby('ProductCD')['isFraud'].value_counts().unstack().plot(kind='bar')

# As Expected, the proportion of Fraudulent Transactions are higher in Product Code of W and C and are least or negligible with

# Other Product Codes.
test['ProductCD'].value_counts(dropna=False)
card = [c for c in train.columns if c.startswith("card")]

card_columns = train[card]

card_columns.head()
for col in card_columns:

    print ("Number of Unique Values in {} column are".format(col),card_columns[col].nunique())
for col in card_columns:

    print ("Number of Unique Values in {} column in test dataset are".format(col),test[col].nunique())
for col in card_columns:

    print ("Number of Missing Values in {} column in the training set are".format(col),card_columns[col].isna().sum())
for col in card_columns:

    print ("Number of Missing Values in {} column in the test set are".format(col),test[col].isna().sum())
train['card1'] = train['card1'].astype('category')

train['card1'] = train['card1'].cat.codes



train['card2'].fillna(train['card2'].value_counts(dropna=False).index[0],inplace=True)

train['card2'] = train['card2'].astype('category')

train['card2'] = train['card2'].cat.codes



train['card3'].fillna(train['card3'].value_counts(dropna=False).index[0],inplace=True)

train['card3'] = train['card3'].astype('category')

train['card3'] = train['card3'].cat.codes



train['card4'].fillna("Missing",inplace=True)

train['card4'] = train['card4'].astype('category')

train['card4'] = train['card4'].cat.codes



train['card5'].fillna(train['card5'].value_counts(dropna=False).index[0],inplace=True)

train['card5'] = train['card5'].astype('category')

train['card5'] = train['card5'].cat.codes



train['card6'].fillna("Missing",inplace=True)

train['card6'] = train['card6'].astype('category')

train['card6'] = train['card6'].cat.codes
test['card1'] = test['card1'].astype('category')

test['card1'] = test['card1'].cat.codes



test['card2'].fillna(test['card2'].value_counts(dropna=False).index[0],inplace=True)

test['card2'] = test['card2'].astype('category')

test['card2'] = test['card2'].cat.codes



test['card3'].fillna(test['card3'].value_counts(dropna=False).index[0],inplace=True)

test['card3'] = test['card3'].astype('category')

test['card3'] = test['card3'].cat.codes



test['card4'].fillna("Missing",inplace=True)

test['card4'] = test['card4'].astype('category')

test['card4'] = test['card4'].cat.codes



test['card5'].fillna(test['card5'].value_counts(dropna=False).index[0],inplace=True)

test['card5'] = test['card5'].astype('category')

test['card5'] = test['card5'].cat.codes



test['card6'].fillna("Missing",inplace=True)

test['card6'] = test['card6'].astype('category')

test['card6'] = test['card6'].cat.codes
train.groupby('card4')['isFraud'].value_counts().unstack()
train.groupby('card4')['isFraud'].value_counts().unstack().plot(kind='bar')
train.groupby('card6')['isFraud'].value_counts().unstack()
train.groupby('card6')['isFraud'].value_counts().unstack().plot(kind='bar')
train[train['addr1'].isna()][['addr1','addr2']].head()

# Rows with addr1 as missing have addr2 as missing too.
test[test['addr1'].isna()][['addr1','addr2']].head()

# Rows with addr1 as missing have addr2 as missing too.
print ("Percentage Of Missing Values in {} Column is:".format("addr1"),train['addr1'].isna().sum()/len(train))

print ("Number Of Unique Values in {} Column are:".format("addr1"),train['addr1'].nunique())

print ("Percentage Of Missing Values in {} Column is:".format("addr2"),train['addr2'].isna().sum()/len(train))

print ("Number Of Unique Values in {} Column are:".format("addr2"),train['addr2'].nunique())
print ("Percentage Of Missing Values in {} Column is:".format("addr1"),test['addr1'].isna().sum()/len(test))

print ("Number Of Unique Values in {} Column are:".format("addr1"),test['addr1'].nunique())

print ("Percentage Of Missing Values in {} Column is:".format("addr2"),test['addr2'].isna().sum()/len(test))

print ("Number Of Unique Values in {} Column are:".format("addr2"),test['addr2'].nunique())
train['addr1'].value_counts(dropna=False).head()

# Missing values constitute the Highest number, so creating a new column addr1_Missing indicating Missing values in the column.
train['addr2'].value_counts(dropna=False).head()
test['addr1'].value_counts(dropna=False).head()
test['addr2'].value_counts(dropna=False).head()
train['addr1_Missing'] = train['addr1'].isna()

train['addr1'].fillna("Missing",inplace=True)

train['addr1'] = train['addr1'].astype('category').cat.codes



train['addr2_Missing'] = train['addr2'].isna()

train['addr2'].fillna("Missing",inplace=True)

train['addr2'] = train['addr2'].astype('category').cat.codes
test['addr1_Missing'] = test['addr1'].isna()

test['addr1'].fillna("Missing",inplace=True)

test['addr1'] = test['addr1'].astype('category').cat.codes



test['addr2_Missing'] = test['addr2'].isna()

test['addr2'].fillna("Missing",inplace=True)

test['addr2'] = test['addr2'].astype('category').cat.codes
train[['P_emaildomain','R_emaildomain']].head()
test[['P_emaildomain','R_emaildomain']].head()
print ("Percentage Of Missing Values in {} Column is:".format("P_emaildomain"),train['P_emaildomain'].isna().sum()/len(train))

print ("Number Of Unique Values in {} Column are:".format("P_emaildomain"),train['P_emaildomain'].nunique())

print ("Percentage Of Missing Values in {} Column is:".format("R_emaildomain"),train['R_emaildomain'].isna().sum()/len(train))

print ("Number Of Unique Values in {} Column are:".format("R_emaildomain"),train['R_emaildomain'].nunique())
print ("Percentage Of Missing Values in {} Column is:".format("P_emaildomain"),test['P_emaildomain'].isna().sum()/len(test))

print ("Number Of Unique Values in {} Column are:".format("P_emaildomain"),test['P_emaildomain'].nunique())

print ("Percentage Of Missing Values in {} Column is:".format("R_emaildomain"),test['R_emaildomain'].isna().sum()/len(test))

print ("Number Of Unique Values in {} Column are:".format("R_emaildomain"),test['R_emaildomain'].nunique())
# Since the R_emaildomain in both Training and Testing datasets has higher percentage of Missing values, it makes sense to delete

train.drop("R_emaildomain",axis=1,inplace=True)

test.drop("R_emaildomain",axis=1,inplace=True)
train['P_emaildomain'].value_counts(dropna=False).head()
test['P_emaildomain'].value_counts(dropna=False).head()
# Since there are lot of missing values, it makes sense to create an is_missing columnn with respect to P_emaildomain variable.

train['P_email_Missing'] = train['P_emaildomain'].isna()

train['P_emaildomain'].fillna("Missing",inplace=True)

train['P_emaildomain'] = train['P_emaildomain'].astype('category').cat.codes
test['P_email_Missing'] = test['P_emaildomain'].isna()

test['P_emaildomain'].fillna("Missing",inplace=True)

test['P_emaildomain'] = test['P_emaildomain'].astype('category').cat.codes
train.groupby('P_emaildomain')['isFraud'].value_counts().unstack().plot(kind='bar',figsize=(14,6))
m_cols = [col for col in train.columns if col.startswith("M")]

train[m_cols].head()
test[m_cols].head()
for col in m_cols:

    print ("Number of Unique Values in {} Column are:".format(col),train[col].nunique())

    print ("Percentage of Missing Values in {} Column is:".format(col),train[col].isna().sum()/len(train))

    print ("--------------------------------------------------------------------------")
for col in m_cols:

    print ("Number of Unique Values in {} Column are:".format(col),test[col].nunique())

    print ("Percentage of Missing Values in {} Column is:".format(col),test[col].isna().sum()/len(test))

    print ("--------------------------------------------------------------------------")
train[train['M1'].isna()][['M1','M2','M3']]

# The number of Missing values across columns M1,M2 and M3 are the same.
train[train['M7'].isna()][['M7','M8','M9']]

# The number of Missing values across columns M7,M8 and M9 are the same.
train['M4'].value_counts(dropna=False)
train['M1'].fillna("Missing",inplace=True)

train['M1'] = train['M1'].astype('category').cat.codes



train['M2'].fillna("Missing",inplace=True)

train['M2'] = train['M2'].astype('category').cat.codes



train['M3'].fillna("Missing",inplace=True)

train['M3'] = train['M3'].astype('category').cat.codes



train['M4'].fillna("Missing",inplace=True)

train['M4'] = train['M4'].astype('category').cat.codes



train['M5'].fillna("Missing",inplace=True)

train['M5'] = train['M5'].astype('category').cat.codes



train['M6'].fillna("Missing",inplace=True)

train['M6'] = train['M6'].astype('category').cat.codes



train['M7'].fillna("Missing",inplace=True)

train['M7'] = train['M7'].astype('category').cat.codes



train['M8'].fillna("Missing",inplace=True)

train['M8'] = train['M8'].astype('category').cat.codes



train['M9'].fillna("Missing",inplace=True)

train['M9'] = train['M9'].astype('category').cat.codes
test['M1'].fillna("Missing",inplace=True)

test['M1'] = test['M1'].astype('category').cat.codes



test['M2'].fillna("Missing",inplace=True)

test['M2'] = test['M2'].astype('category').cat.codes



test['M3'].fillna("Missing",inplace=True)

test['M3'] = test['M3'].astype('category').cat.codes



test['M4'].fillna("Missing",inplace=True)

test['M4'] = test['M4'].astype('category').cat.codes



test['M5'].fillna("Missing",inplace=True)

test['M5'] = test['M5'].astype('category').cat.codes



test['M6'].fillna("Missing",inplace=True)

test['M6'] = test['M6'].astype('category').cat.codes



test['M7'].fillna("Missing",inplace=True)

test['M7'] = test['M7'].astype('category').cat.codes



test['M8'].fillna("Missing",inplace=True)

test['M8'] = test['M8'].astype('category').cat.codes



test['M9'].fillna("Missing",inplace=True)

test['M9'] = test['M9'].astype('category').cat.codes
train['DeviceType'].value_counts(dropna=False)
test['DeviceType'].value_counts(dropna=False)
train['DeviceType_Missing'] = train['DeviceType'].isna()

train['DeviceType'].fillna("Missing",inplace=True)
test['DeviceType_Missing'] = test['DeviceType'].isna()

test['DeviceType'].fillna("Missing",inplace=True)
train.groupby('DeviceType')['isFraud'].mean()

# Although the Number of Mobile devices are less in number, the percentage of Fraudulent Transactions is highest.
train.groupby('DeviceType')['isFraud'].value_counts().unstack().plot(kind='bar',figsize=(10,6))

plt.ylabel("Count")

plt.title("Fraudulent Transactions Grouped By Device Type")
print ("Number Of Unique Values in {} Variable are:".format("DeviceInfo"),train['DeviceInfo'].nunique())
train['DeviceInfo'].value_counts(dropna=False).head()
print ("Number Of Unique Values in {} Variable are:".format("DeviceInfo"),test['DeviceInfo'].nunique())
train['DeviceInfo_Missing'] = train['DeviceInfo'].isna()

train['DeviceInfo'].fillna("Missing",inplace=True)

train['DeviceInfo'] = train['DeviceInfo'].astype('category').cat.codes
test['DeviceInfo_Missing'] = test['DeviceInfo'].isna()

test['DeviceInfo'].fillna("Missing",inplace=True)

test['DeviceInfo'] = test['DeviceInfo'].astype('category').cat.codes
cols = [col for col in train.columns if col.startswith("id")]

id_cols = train[cols]

id_cols.head()
test[cols].head()
for col in cols:

    print ("Number of Unique Values in {} Column are:".format(col),train[col].nunique())

    print ("Percentage of Missing Values in {} Column is:".format(col),train[col].isna().sum()/len(train))

    print ("--------------------------------------------------------------------------")
for col in cols:

    print ("Number of Unique Values in {} Column are:".format(col),test[col].nunique())

    print ("Percentage of Missing Values in {} Column is:".format(col),test[col].isna().sum()/len(test))

    print ("--------------------------------------------------------------------------")
common_id = []

for col in id_cols:

    if (train[col].isna().sum()/len(train) > 0.75) & (test[col].isna().sum()/len(test) > 0.75):

        common_id.append(col)

print (common_id)
train.drop(common_id,axis=1,inplace=True)

test.drop(common_id,axis=1,inplace=True)
cols = [col for col in train.columns if col.startswith("id")]

id_cols = train[cols]

id_cols.head()
for col in cols:

    train[col+"_Missing"] = train[col].isna()

    

for col in cols:

    test[col+"_Missing"] = test[col].isna()
for col in cols:

    train[col].fillna("Missing",inplace=True)



for col in cols:

    test[col].fillna("Missing",inplace=True)
for col in cols:

    train[col] = train[col].astype('category').cat.codes



for col in cols:

    test[col] = test[col].astype('category').cat.codes
train['isFraud'].value_counts(dropna=False)/len(train)
sns.countplot(train['isFraud'])

plt.xlabel("isFraud")

plt.ylabel("Count")

plt.title("Distribution of isFraud(Dependent) Variable")
train['dist1'].describe()
test['dist1'].describe()
train['dist1'].fillna(train['dist1'].median(),inplace=True)

test['dist1'].fillna(test['dist1'].median(),inplace=True)
train[train['TransactionAmt'].isna()]

# No Missing Values Here
test[test['TransactionAmt'].isna()]
train['TransactionAmt'].describe()
test['TransactionAmt'].describe()
train.groupby('isFraud')['TransactionAmt'].agg({'min','max','mean','median'})
sns.distplot(train['TransactionAmt'],kde=False)

plt.xlabel("Transaction Amounts")

plt.ylabel("Distribution")

plt.title("Distribution Plot of Transaction Amount variable")
sns.distplot(np.log(train['TransactionAmt']),kde=False)

plt.xlabel("Transaction Amounts")

plt.ylabel("Distribution")

plt.title("Distribution Plot of 'Log of Transaction Amount' variable")

# More or less this looks like a normal distribution.
sns.distplot(np.log(test['TransactionAmt']),kde=False)

plt.xlabel("Transaction Amounts")

plt.ylabel("Distribution")

plt.title("Distribution Plot of 'Log of Transaction Amount' variable")

# More or less this looks like a normal distribution.
cols = [col for col in train.columns if col.startswith("C")]

c_cols = train[cols]

c_cols.head()
test[cols].head()
for col in cols:

    print ("Number of Unique Values in {} Column are:".format(col),train[col].nunique())

    print ("Percentage of Missing Values in {} Column is:".format(col),train[col].isna().sum()/len(train))

    print ("--------------------------------------------------------------------------")
for col in cols:

    print ("Number of Unique Values in {} Column are:".format(col),test[col].nunique())

    print ("Percentage of Missing Values in {} Column is:".format(col),test[col].isna().sum()/len(test))

    print ("--------------------------------------------------------------------------")
for col in cols:

    test[col].fillna(test[col].value_counts(dropna=False).index[0],inplace=True)
cols = ["D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15"]

d_cols = train[cols]

d_cols.head()
test[cols].head()
for col in cols:

    print ("Number of Unique Values in {} Column are:".format(col),train[col].nunique())

    print ("Percentage of Missing Values in {} Column is:".format(col),train[col].isna().sum()/len(train))

    print ("--------------------------------------------------------------------------")
for col in cols:

    print ("Number of Unique Values in {} Column are:".format(col),test[col].nunique())

    print ("Percentage of Missing Values in {} Column is:".format(col),test[col].isna().sum()/len(test))

    print ("--------------------------------------------------------------------------")
common_d = []

for col in cols:

    if (train[col].isna().sum()/len(train) > 0.8) & (test[col].isna().sum()/len(test) > 0.8):

        common_d.append(col)

print (common_d)    
train.drop(["D7","D8","D12"],axis=1,inplace=True)

test.drop(["D7","D8","D12"],axis=1,inplace=True)
cols = ["D1","D2","D3","D4","D5","D6","D9","D10","D11","D13","D14","D15"]
for col in cols:

    train[col].fillna(0.0,inplace=True)

    

for col in cols:

    test[col].fillna(0.0,inplace=True)
cols = [col for col in train.columns if col[0]=="V"]

v_cols = train[cols]

v_cols.head()
test[cols].head()
for col in cols:

    if train[col].nunique() == 2:

        print ("Unique Values in {} Column in Training Set are:".format(col),train[col].unique())

        print ("Percentage of Missing Values in {} Column in Training Set is:".format(col),train[col].isna().sum()/len(train))

        print ("--------------------------------------------------------------------------")
for col in cols:

    if test[col].nunique() == 2:

        print ("Unique Values in {} Column in Test Set are:".format(col),test[col].unique())

        print ("Percentage of Missing Values in {} Column in Test Set is:".format(col),test[col].isna().sum()/len(test))

        print ("--------------------------------------------------------------------------")
for col in ["V1","V14","V41","V65","V88","V107","V305"]:

    print ("Value Counts in {} Variable:".format(col))

    print (train[col].value_counts(dropna=False))
for col in ["V1","V14","V41","V65","V88","V107","V305"]:

    train[col].fillna(train[col].value_counts(dropna=False).index[0],inplace=True)

for col in ["V1","V14","V41","V65","V88","V107","V305"]:

    test[col].fillna(test[col].value_counts(dropna=False).index[0],inplace=True)
common_v = []

for col in cols:

    if (train[col].isna().sum()/len(train) > 0.8) & (test[col].isna().sum()/len(test) > 0.8):

        common_v.append(col)

print (common_v) 
train.drop(common_v,axis=1,inplace=True)

test.drop(common_v,axis=1,inplace=True)
cols = [col for col in train.columns if col[0]=="V"]

for col in cols:

    train[col].fillna(train[col].median(),inplace=True)

for col in cols:

    test[col].fillna(test[col].median(),inplace=True)
overfit = []

for col in train.columns:

    counts = train[col].value_counts()

    zeros = counts.iloc[0]

    if zeros / len(train) * 100 >99:

        overfit.append(col)

print (len(overfit))

print (overfit)
train.drop(overfit,axis=1,inplace=True)

test.drop(overfit,axis=1,inplace=True)
train.shape, test.shape
%%time

# Let's check the correlation between the variables and eliminate the one's that have high correlation

# Threshold for removing correlated variables

threshold = 0.9



# Absolute value correlation matrix

corr_matrix = train.corr().abs()

corr_matrix.head()
# Upper triangle of correlations

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

upper.head()
# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]



print('There are %d columns to remove.' % (len(to_drop)))

print ("Following columns can be dropped {}".format(to_drop))
train.drop(to_drop,axis=1,inplace=True)

test.drop(to_drop,axis=1,inplace=True)
train.shape, test.shape
X = train[[col for col in train.columns if col!='isFraud']]

y = train['isFraud']
X = pd.get_dummies(X)

test = pd.get_dummies(test)

print (train.shape)

print (test.shape)
X_Train,X_Test,y_Train,y_Test = train_test_split(X,y,test_size=0.25)

print (X_Train.shape)

print (y_Train.shape)

print (X_Test.shape)

print (y_Test.shape)
clf = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',random_state=42,class_weight='balanced',n_jobs=-1,verbose=1)

params = {"max_depth":[3,4,5,6,-1],

          "learning_rate":[0.01,0.05,0.1,0.3],

          "subsample":[0.5,0.7,0.9],

          "colsample_bytree":[0.5,0.7,0.9],

          "reg_alpha":[0.5,1,2],

          "reg_lambda":[0.5,1,2],

          "num_leaves":[7,15,31,63],

          "n_estimators":list(range(50,300,50))}

random_search = RandomizedSearchCV(estimator=clf,param_distributions=params,cv=5,scoring='roc_auc')

random_search.fit(X_Train,y_Train)
"""clf = xgb.XGBClassifier(random_state=42,n_jobs=-1,verbosity=1)

params = {"max_depth":[3,5,7,9],

          "n_estimators":list(range(50,301,50)),

          "learning_rate":[0.01,0.05,0.1,0.3],

          "subsample":[0.5,0.7,0.9],

          "colsample_bytree":[0.5,0.7,0.9],

          "reg_alpha":[0.5,1,2,5],

          "reg_lambda":[0.5,1,2,5]}

random_search = RandomizedSearchCV(estimator=clf,param_distributions=params,cv=5,scoring='roc_auc')

random_search.fit(X_Train,y_Train)"""
random_search.best_estimator_,random_search.best_params_,random_search.best_score_
ser = pd.Series(random_search.best_estimator_.feature_importances_,X_Train.columns).sort_values()

lst = list(ser[ser > 0].index)

print (lst)
X = X[lst]

test = test[lst]



X_Train,X_Test,y_Train,y_Test = train_test_split(X,y,test_size=0.25)

print (X_Train.shape)

print (y_Train.shape)

print (X_Test.shape)

print (y_Test.shape)



clf = lgb.LGBMClassifier(boosting_type='gbdt',objective='binary',random_state=42,class_weight='balanced',n_jobs=-1,verbose=1)

params = {"max_depth":[3,4,5,6,-1],

          "learning_rate":[0.01,0.05,0.1,0.3],

          "subsample":[0.5,0.7,0.9],

          "colsample_bytree":[0.5,0.7,0.9],

          "reg_alpha":[0.5,1,2],

          "reg_lambda":[0.5,1,2],

          "num_leaves":[7,15,31,63],

          "n_estimators":list(range(50,300,50))}

random_search = RandomizedSearchCV(estimator=clf,param_distributions=params,cv=5,scoring='roc_auc')

random_search.fit(X_Train,y_Train)
random_search.best_estimator_,random_search.best_params_,random_search.best_score_
Submission['isFraud'] = random_search.best_estimator_.predict_proba(test)[:,1]

Submission.to_csv("LightGBM.csv",index=None)