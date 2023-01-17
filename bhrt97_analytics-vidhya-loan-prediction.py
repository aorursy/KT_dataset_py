import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train_df = pd.read_csv("/kaggle/input/loan-pred-traincsv/Loan pred_train.csv") 
test_df = pd.read_csv("/kaggle/input/loan-pred-traincsv/Loan Pred_test.csv")
# Loan amount in thousands
train_df['LoanAmount'] = train_df['LoanAmount'] * 1000
test_df['LoanAmount'] = test_df['LoanAmount'] * 1000
# Loan_Amount_Term: Term of loan in months
train_df.head(5)
train_df.shape
train_df.describe().T
test_df.describe().T
test_df.shape
cols_with_missing = [col for col in train_df.columns 
                                 if train_df[col].isnull().any()]
cols_with_missing
train_df.isna().sum()
for col in train_df.columns:
    percent = train_df[col].isna().sum() / len(train_df)
    print('{} \t {}'.format(col,round(percent*100)))
# train_df.dropna(inplace=True)   # give resulting shape (480,13)

# so we are lossing 614-480 = 134 rows if we blindly drop all rows with missing values
train_df.duplicated().sum()
cols_with_missing = [col for col in test_df.columns 
                                 if test_df[col].isnull().any()]
cols_with_missing
train_df.skew()
train_df.kurt()
train_df.dtypes
train_df['CoapplicantIncome'].value_counts()
train_df['Gender'].fillna((train_df['Gender'].mode()[0]),inplace=True)
train_df['Married'].fillna((train_df['Married'].mode()[0]),inplace=True)
train_df['Dependents'].fillna((train_df['Dependents'].mode()[0]),inplace=True)
train_df['Education'].fillna((train_df['Education'].mode()[0]),inplace=True)###############
train_df['Gender'].fillna((train_df['Gender'].mode()[0]),inplace=True)
train_df['Self_Employed'].fillna((train_df['Self_Employed'].mode()[0]),inplace=True)


test_df['Gender'].fillna((train_df['Gender'].mode()[0]),inplace=True)
test_df['Married'].fillna((train_df['Married'].mode()[0]),inplace=True)
test_df['Dependents'].fillna((train_df['Dependents'].mode()[0]),inplace=True)
test_df['Education'].fillna((train_df['Education'].mode()[0]),inplace=True)################
test_df['Gender'].fillna((train_df['Gender'].mode()[0]),inplace=True)
test_df['Self_Employed'].fillna((train_df['Self_Employed'].mode()[0]),inplace=True)

# replacing '+' from Dependent column
train_df['Dependents']=train_df['Dependents'].apply(lambda x:str(x).replace('+','')if '+' in str(x) else str(x))
train_df['Dependents']=train_df['Dependents'].apply(lambda x:int(x))

test_df['Dependents']=test_df['Dependents'].apply(lambda x:str(x).replace('+','')if '+' in str(x) else str(x))
test_df['Dependents']=test_df['Dependents'].apply(lambda x:int(x))
train_df.isna().sum()
print(train_df['LoanAmount'].median())
print(train_df['LoanAmount'].mode())
print(train_df['LoanAmount'].mean())
print(" ")
print(train_df['Loan_Amount_Term'].median())
print(train_df['Loan_Amount_Term'].mode())
print(train_df['Loan_Amount_Term'].mean())
print(" ")
print(train_df['Credit_History'].median())
print(train_df['Credit_History'].mode())
print(train_df['Credit_History'].mean())
train_df['LoanAmount'].fillna((train_df['LoanAmount'].median()),inplace=True)
train_df['Loan_Amount_Term'].fillna((train_df['Loan_Amount_Term'].median()),inplace=True)
# train_df['Credit_History'].fillna((train_df['Credit_History'].median()),inplace=True)

test_df['LoanAmount'].fillna((test_df['LoanAmount'].median()),inplace=True)
test_df['Loan_Amount_Term'].fillna((test_df['Loan_Amount_Term'].median()),inplace=True)
# test_df['Credit_History'].fillna((test_df['Credit_History'].median()),inplace=True)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
train_df['Loan_Status'] = encoder.fit_transform(train_df['Loan_Status'])
#filling credit_history where loan status was approved
train_df['Credit_History'] = np.where(((train_df['Credit_History'].isnull()) & (train_df['Loan_Status'] ==1)),
                                   1,train_df['Credit_History'])

#filling credit_history based on where loan status was declined
train_df['Credit_History'] = np.where(((train_df['Credit_History'].isnull()) & (train_df['Loan_Status'] ==0)),
                                   0,train_df['Credit_History'])
#filling credit_history where loan status was approved
# test_df['Credit_History'] = np.where(((test_df['Credit_History'].isnull()) & (train_df['Loan_Status'] ==1)),
#                                    1,test_df['Credit_History'])

#filling credit_history based on where loan status was declined
# test_df['Credit_History'] = np.where(((test_df['Credit_History'].isnull()) & (test_df['Loan_Status'] ==0)),
#                                    0,test_df['Credit_History'])
train_df['Credit_History'].value_counts()
train_df.drop("Loan_ID",axis=1,inplace=True)
test_df.drop("Loan_ID",axis=1,inplace=True)
train_df.isna().sum()
test_df.isna().sum()


cat_list = ['Gender','Married','Education','Self_Employed','Property_Area']

for i in cat_list:
    le = LabelEncoder()
    train_df[i] = train_df[i].astype('str')
    train_df[i] = le.fit_transform(train_df[i])
    test_df[i] = test_df[i].astype('str')
    test_df[i] = le.fit_transform(test_df[i])
    
le = LabelEncoder()
train_df['Loan_Status'] = le.fit_transform(train_df['Loan_Status'])
#Log transfromations
train_df['LoanAmount'] = np.log1p(train_df['LoanAmount'])

#Log transforming features
train_df['ApplicantIncome'] = np.log1p(train_df['ApplicantIncome'])
train_df['CoapplicantIncome'] = np.log1p(train_df['CoapplicantIncome'])

#coapplicant income and applicant income both serves as determinants for loan status
#log transformation

train_df['total_income'] = train_df['ApplicantIncome'] + train_df['CoapplicantIncome']
train_df['total_income'] = np.log1p(train_df['total_income'])

#Log transformation 
train_df['Ratio of LoanAmt :Total_Income'] = train_df['LoanAmount'] / train_df['total_income']
train_df['Ratio of LoanAmt :Total_Income'] = np.log1p(train_df['Ratio of LoanAmt :Total_Income'])
#Log transfromations
test_df['LoanAmount'] = np.log1p(test_df['LoanAmount'])

#Log transforming features
test_df['ApplicantIncome'] = np.log1p(test_df['ApplicantIncome'])
test_df['CoapplicantIncome'] = np.log1p(test_df['CoapplicantIncome'])

#coapplicant income and applicant income both serves as determinants for loan status
#log transformation

test_df['total_income'] = test_df['ApplicantIncome'] + test_df['CoapplicantIncome']
test_df['total_income'] = np.log1p(test_df['total_income'])

#Log transformation 
test_df['Ratio of LoanAmt :Total_Income'] = test_df['LoanAmount'] / test_df['total_income']
test_df['Ratio of LoanAmt :Total_Income'] = np.log1p(test_df['Ratio of LoanAmt :Total_Income'])
X = train_df.drop("Loan_Status",axis=1)
y =  train_df.Loan_Status
from sklearn.model_selection import train_test_split as split

X_train, X_test, y_train, y_test = split(X,y, test_size=0.1, random_state=111)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# X_train_scaled.skew()
# X_train_pca.kurt()
from sklearn.decomposition import PCA
# score with all 6 the features
pca = PCA()
temp_X_train= X_train
temp_X_train = pca.fit_transform(temp_X_train)
pca.explained_variance_ratio_
plt.figure(figsize=(8,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,12,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression(max_iter = 200)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

print('Logistic Regression accuracy : {}'. format(accuracy_score(y_test, y_pred)))
X_train.shape
pca = PCA(n_components=6)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression(max_iter = 200)
logreg.fit(X_train_pca, y_train)
y_pred = logreg.predict(X_test_pca)

print('Logistic Regression accuracy : {}'. format(accuracy_score(y_test, y_pred)))
from xgboost import XGBClassifier

clf= XGBClassifier(learning_rate=0.05, n_estimators=206, max_depth=2,
                        min_child_weight=4, 
                         seed=27)

clf.fit(X_train, y_train)

# y_pred = logreg.predict(X_test)
y_pred = clf.predict(X_test)
print('XGboost accuracy score {}'. format(accuracy_score(y_test, y_pred)))
from catboost import CatBoostClassifier

clf= CatBoostClassifier(learning_rate=0.05, n_estimators=200, max_depth=3)

clf.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
y_pred = clf.predict(X_test)
print('catboost accuracy score{0:0.4f}'. format(accuracy_score(y_test, y_pred)))
from lightgbm import LGBMClassifier

clf= LGBMClassifier(boosting_type='gbdt',
    num_leaves=20,
    max_depth=2,
    learning_rate=0.005,
    n_estimators=150,
#     subsample=0.9,
                   )

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print('LGBMClassifier accuracy: {}'. format(accuracy_score(y_test, y_pred)))
clf.score(X_train, y_train)
X_train, X_test, y_train, y_test = split(X,y, test_size=0.3, random_state=111)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from xgboost import XGBClassifier

clf= XGBClassifier(learning_rate=0.05, n_estimators=206, max_depth=2,
                        min_child_weight=4, 
                         seed=27)

clf.fit(X_train, y_train)

# # y_pred = logreg.predict(X_test)
# y_pred = clf.predict(X_test)
# print('XGboost accuracy score {}'. format(accuracy_score(y_test, y_pred)))
result = clf.predict(test_df)
test_id = pd.read_csv("/kaggle/input/loan-pred-traincsv/Loan Pred_test.csv")
test_id = test_id.Loan_ID
result = pd.DataFrame(result)
test_id = pd.DataFrame(test_id)
submission = pd.merge(test_id,result,left_index=True,right_index=True)
submission.head(1)
submission.rename(columns={0:'Loan_Status'},inplace=True)
submission.head(1)
submission.to_csv('Loan.csv',index=False)
submission['Loan_Status'].value_counts()

