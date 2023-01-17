import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import classification_report, roc_auc_score

from sklearn.model_selection import train_test_split 

from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier

from lightgbm import LGBMClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler







import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings 

warnings.filterwarnings('ignore')

pd.set_option('max_columns', 20)
train = pd.read_csv('/kaggle/input/janatahack-crosssell-prediction/train.csv',index_col=0)

test = pd.read_csv('/kaggle/input/janatahack-crosssell-prediction/test.csv',index_col=0)

sample_submission = pd.read_csv('/kaggle/input/janatahack-crosssell-prediction/sample_submission.csv',index_col=0)
train[['Driving_License','Previously_Insured','Policy_Sales_Channel','Region_Code']] = train[['Driving_License','Previously_Insured','Policy_Sales_Channel','Region_Code']].astype('object')

train['Response'] = train['Response'].astype('object')



test[['Driving_License','Previously_Insured','Policy_Sales_Channel','Region_Code']] = test[['Driving_License','Previously_Insured','Policy_Sales_Channel','Region_Code']].astype('object')
sns.distplot(train['Annual_Premium']);
premium_median = train['Annual_Premium'].median()

train['Annual_Premium'] = np.where(train['Annual_Premium']> 100000.000000, premium_median, train['Annual_Premium'])

sns.distplot(train['Annual_Premium']);
premium_median_test = test['Annual_Premium'].median()

test['Annual_Premium'] = np.where(test['Annual_Premium']> 100000.000000, premium_median_test, test['Annual_Premium'])

sns.distplot(test['Annual_Premium']);
plt.figure(figsize=(20,6))

sns.countplot(x='Region_Code',data=train,hue='Response');
sns.countplot(x='Driving_License',data=train,hue='Response');
sns.countplot(x='Vehicle_Damage',data=train,hue='Response');
# channel = train['Policy_Sales_Channel'].value_counts()

# pct_80 = train.shape[0]*0.80

# channel.cumsum()[channel.cumsum()<pct_80]



# train[~train['Policy_Sales_Channel'].isin([152,26,124])]['Policy_Sales_Channel'] = 999



# channel.cumsum().iloc[:10]

#every channel other than 152, 26 and 124 has to be coded as 999 i.e. other
train['Vehicle_Age_Damage'] = train['Vehicle_Age'] + '_' + train['Vehicle_Damage']
test['Vehicle_Age_Damage'] = test['Vehicle_Age'] + '_' + test['Vehicle_Damage']
sns.countplot(x='Gender',data=train,hue='Response');
sns.countplot(x='Previously_Insured',data=train,hue='Response');
train.head()
X = train.drop('Response',axis=1)

y = train['Response'].values
#Scaling numeric variables



sc = StandardScaler()

X[X.select_dtypes(exclude='object').columns.to_list()] = sc.fit_transform(X.select_dtypes(exclude='object'))
test[test.select_dtypes(exclude='object').columns.to_list()] = sc.transform(test.select_dtypes(exclude='object'))
X.head()
test.head()
#Encoding categorical variables



X = pd.get_dummies(X,drop_first=True)

test = pd.get_dummies(test,drop_first=True)
le = LabelEncoder()

y = le.fit_transform(y)
clf = LGBMClassifier(n_estimators=550,

                     learning_rate=0.03,

                     min_child_samples=40,

                     random_state=1,

                     colsample_bytree=0.5,

                     reg_alpha=2,

                     reg_lambda=2)



clf.fit(X, y, verbose=50,eval_metric = 'auc')
lgb_pred = clf.predict_proba(X)[:,1]

roc_auc_score(y,lgb_pred)
print(test.shape,X.shape)



missing_cols = set(X.columns) - set(test.columns)

for c in missing_cols:

    test[c] = 0



print(test.shape,X.shape)



#keeping the order of columns same for X and test

test = test[X.columns]
test_pred = clf.predict_proba(test)[:,1]
sample_submission['Response'] = test_pred

sample_submission.to_csv('Submission_v5.csv')
sample_submission.head()