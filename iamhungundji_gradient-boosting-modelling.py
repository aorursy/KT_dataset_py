import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_csv('../input/janatahack-machine-learning-for-banking/train_fNxu4vz.csv')

test = pd.read_csv('../input/janatahack-machine-learning-for-banking/test_fjtUOL8.csv')
train.head()
train.isnull().sum()
train.shape, test.shape
combine = train.append(test)

combine.shape
combine.columns
combine['Annual_Income'].describe()
combine['Annual_Income'] = np.log(combine['Annual_Income'])

combine['Annual_Income'].fillna(combine['Annual_Income'].mean(), inplace=True)

combine['Annual_Income'].describe()
combine['Debt_To_Income'].describe()
combine['Debt_To_Income'] = np.log1p(combine['Debt_To_Income'])

combine['Debt_To_Income'].describe()
combine['Gender'].value_counts()
combine['Home_Owner'].value_counts()
combine['Home_Owner'].fillna('Unknown', inplace=True)

combine['Home_Owner'].value_counts()
combine['Income_Verified'].value_counts()
combine['Income_Verified'] = combine['Income_Verified'].replace('VERIFIED - income', 'Verf_inc')

combine['Income_Verified'] = combine['Income_Verified'].replace('VERIFIED - income source', 'Verf_inc_src')

combine['Income_Verified'] = combine['Income_Verified'].replace('not verified', 'Not_verf')

combine['Income_Verified'].value_counts()
combine['Inquiries_Last_6Mo'].value_counts()
combine['Length_Employed'].value_counts()
combine['Length_Employed'] = combine['Length_Employed'].replace('10+ years', 10)

combine['Length_Employed'] = combine['Length_Employed'].replace('2 years', 2)

combine['Length_Employed'] = combine['Length_Employed'].replace('3 years', 3)

combine['Length_Employed'] = combine['Length_Employed'].replace('< 1 year', 0)

combine['Length_Employed'] = combine['Length_Employed'].replace('5 years', 5)

combine['Length_Employed'] = combine['Length_Employed'].replace('1 year', 1)

combine['Length_Employed'] = combine['Length_Employed'].replace('4 years', 4)

combine['Length_Employed'] = combine['Length_Employed'].replace('7 years', 7)

combine['Length_Employed'] = combine['Length_Employed'].replace('6 years', 6)

combine['Length_Employed'] = combine['Length_Employed'].replace('8 years', 8)

combine['Length_Employed'] = combine['Length_Employed'].replace('9 years', 9)

combine['Length_Employed'].fillna(-1, inplace=True)

combine['Length_Employed'] = combine['Length_Employed'].astype('int')

combine['Length_Employed'].value_counts()
combine['Loan_Amount_Requested'] = combine['Loan_Amount_Requested'].str.replace(',','').astype('int')

combine['Loan_Amount_Requested'].describe()
combine['Loan_Amount_Requested'] = np.log(combine['Loan_Amount_Requested'])

combine['Loan_Amount_Requested'].describe()
combine['Months_Since_Deliquency'].describe()
combine['Months_Since_Deliquency'].fillna(-1, inplace=True)

bins= [-1, 0, 30, 60, 182]

labels = ['Unknown', 'Month_1', 'Month_2', 'Month_3Plus']

combine['Months_Since_Deliquency'] = pd.cut(combine['Months_Since_Deliquency'], bins=bins, labels=labels, right=False)

combine['Months_Since_Deliquency'].value_counts()
combine['Number_Open_Accounts'].describe()
bins= [0, 10, 20, 86]

labels = ['Acc_Tier_1', 'Acc_Tier_2', 'Acc_Tier_3']

combine['Number_Open_Accounts'] = pd.cut(combine['Number_Open_Accounts'], bins=bins, labels=labels, right=False)

combine['Number_Open_Accounts'].value_counts()
combine['Purpose_Of_Loan'].value_counts()
combine['Total_Accounts'].describe()
bins= [2, 12, 22, 32, 159]

labels = ['Tot_Acc_1', 'Tot_Acc_2', 'Tot_Acc_3', 'Tot_Acc_4']

combine['Total_Accounts'] = pd.cut(combine['Total_Accounts'], bins=bins, labels=labels, right=False)

combine['Total_Accounts'].value_counts()
combine.isnull().sum()
combine.dtypes
train_cleaned = combine[combine['Interest_Rate'].isnull()!=True].drop(['Loan_ID'], axis=1)

train_cleaned.head()
Gender = pd.crosstab(train_cleaned['Gender'], train_cleaned['Interest_Rate'])

Home_Owner = pd.crosstab(train_cleaned['Home_Owner'], train_cleaned['Interest_Rate'])

Income_Verified = pd.crosstab(train_cleaned['Income_Verified'], train_cleaned['Interest_Rate'])

Inquiries_Last_6Mo = pd.crosstab(train_cleaned['Inquiries_Last_6Mo'], train_cleaned['Interest_Rate'])

Length_Employed = pd.crosstab(train_cleaned['Length_Employed'], train_cleaned['Interest_Rate'])

Months_Since_Deliquency = pd.crosstab(train_cleaned['Months_Since_Deliquency'], train_cleaned['Interest_Rate'])

Number_Open_Accounts = pd.crosstab(train_cleaned['Number_Open_Accounts'], train_cleaned['Interest_Rate'])

Purpose_Of_Loan = pd.crosstab(train_cleaned['Purpose_Of_Loan'], train_cleaned['Interest_Rate'])

Total_Accounts = pd.crosstab(train_cleaned['Total_Accounts'], train_cleaned['Interest_Rate'])





Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(4,4))

Home_Owner.div(Home_Owner.sum(1).astype(float), axis=0).plot(kind="bar")

Income_Verified.div(Income_Verified.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(4,4))

Inquiries_Last_6Mo.div(Inquiries_Last_6Mo.sum(1).astype(float), axis=0).plot(kind="bar")

Length_Employed.div(Length_Employed.sum(1).astype(float), axis=0).plot(kind="bar")

Months_Since_Deliquency.div(Months_Since_Deliquency.sum(1).astype(float), axis=0).plot(kind="bar")

Number_Open_Accounts.div(Number_Open_Accounts.sum(1).astype(float), axis=0).plot(kind="bar")

Purpose_Of_Loan.div(Purpose_Of_Loan.sum(1).astype(float), axis=0).plot(kind="bar")

Total_Accounts.div(Total_Accounts.sum(1).astype(float), axis=0).plot(kind="bar")



plt.show()
matrix = train_cleaned.corr() 

f, ax = plt.subplots(figsize=(9, 6))

sns.heatmap(matrix, vmax=.8, square=True, annot=True)
combine = pd.get_dummies(combine)

combine.shape
X = combine[combine['Interest_Rate'].isnull()!=True].drop(['Loan_ID','Interest_Rate'], axis=1)

y = combine[combine['Interest_Rate'].isnull()!=True]['Interest_Rate']



X_test = combine[combine['Interest_Rate'].isnull()==True].drop(['Loan_ID','Interest_Rate'], axis=1)



X.shape, y.shape, X_test.shape
param = {

    'bagging_freq': 5,

    'bagging_fraction': 0.5,

    'boost': 'gbdt',

    'feature_fraction': 0.7,

    'learning_rate': 0.005,

    'num_class':4,

    'metric':'multi_logloss',

    'max_depth': 8,  

    'num_leaves': 70,

    'min_data_in_leaf':40,

    'objective': 'multiclass',

    'scale_pos_weight':1,

    'verbosity': 1,

    'device':'gpu'

}
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix, f1_score



features = X.columns



folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1048)



pred_test = np.zeros((len(X_test), 4))



feature_importance_df = pd.DataFrame()



for fold_, (trn_idx, val_idx) in enumerate(folds.split(X.values, y.values)):

    print("Fold {}".format(fold_))

    train_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])

    val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])



    num_round = 1000000

    classifier = lgb.train(param, train_data, num_round, valid_sets = [train_data, val_data], 

                    verbose_eval=1000, early_stopping_rounds = 1000)

    pred_y = np.argmax(classifier.predict(X.iloc[val_idx], num_iteration=classifier.best_iteration), axis=1)

    

    print("CV score: ", f1_score(pred_y, y.iloc[val_idx], average='weighted'))

    print(confusion_matrix(pred_y, y.iloc[val_idx]))

    

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = classifier.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    

    pred_test += classifier.predict(X_test, num_iteration=classifier.best_iteration) / folds.n_splits
all_feat = feature_importance_df[["Feature",

                                  "importance"]].groupby("Feature").mean().sort_values(by="importance", 

                                                                                           ascending=False)

all_feat.reset_index(inplace=True)

important_feat = list(all_feat['Feature'])

all_feat
df = X[important_feat]

corr_matrix = df.corr().abs()



upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))



high_cor = [column for column in upper.columns if any(upper[column] > 0.98)]

print(len(high_cor))

print(high_cor)
features = [i for i in important_feat if i not in high_cor]

print(len(features))

print(features)
p_test = np.argmax(pred_test, axis=1)
submission = pd.DataFrame()

submission['Loan_ID'] = test['Loan_ID']

submission['Interest_Rate'] = p_test.astype('int')

submission.head()
submission.to_csv('submission.csv', index=False)

submission['Interest_Rate'].value_counts()