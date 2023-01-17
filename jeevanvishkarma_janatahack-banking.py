import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
train = pd.read_csv('../input/janatahack-machine-learning-for-banking/train_fNxu4vz.csv')
test = pd.read_csv('../input/janatahack-machine-learning-for-banking/test_fjtUOL8.csv')
# combining both data sets
df = pd.concat([train,test], axis = 0)

# Null values
df.isnull().sum()


# employee lenght

df['Length_Employed'].replace({'< 1 year' : '0','1 year': '1','10+ years' : '10'}, inplace = True)
df['Length_Employed'] = df['Length_Employed'].str.replace('years','')

df['Length_Employed'].fillna(-1, inplace = True)
df['Length_Employed'] = df['Length_Employed'].astype(int)

df['Length_Employed'].value_counts()
sns.distplot(df['Length_Employed'])
# HOme owned

df.Home_Owner  = df.Home_Owner.map(lambda x : 3 if x == 'Own' else 2 if x == 'Mortgage'
                                   else 1 if x == 'Rent' else   -1)
sns.countplot(df.Home_Owner)


# loan amount
df.Loan_Amount_Requested = df.Loan_Amount_Requested.str.replace(',','').astype(int)
sns.distplot(df.Loan_Amount_Requested)


# cuberoot transformation is used for positively skewed varible loan amount
df.Loan_Amount_Requested = np.cbrt(df.Loan_Amount_Requested)
sns.distplot(df.Loan_Amount_Requested)

# income verified

sns.countplot(df.Income_Verified)
df.Income_Verified = df.Income_Verified.map(lambda x : 0 if x == 'not verified'
                                                else 3 if x == 'VERIFIED - income' else 2)
# Purpose
sns.countplot(df.Purpose_Of_Loan)
df.Purpose_Of_Loan.value_counts()
pd.crosstab(df.Purpose_Of_Loan, df.Interest_Rate).plot.bar()

df.Purpose_Of_Loan = df.Purpose_Of_Loan.map(lambda x : 5 if x == 'debt_consolidation'
                                            else 4 if x == 'credit_card'
                                            else 3 if x == 'home_improvement'
                                            else 2 if x == 'other'
                                            else -1)
df.Purpose_Of_Loan.astype(int)
# debt is already normaly distributed

df.Debt_To_Income.describe()

sns.distplot(df.Debt_To_Income)
# Inquiries_Last_6Mo
df.Inquiries_Last_6Mo.value_counts()

df.Inquiries_Last_6Mo = df.Inquiries_Last_6Mo.map(lambda x : 3 if x >= 3 else x)
sns.countplot(df.Inquiries_Last_6Mo)

#Months_Since_Deliquency

sns.distplot(df.Months_Since_Deliquency)

df.Months_Since_Deliquency.describe()

# Inter quantile range is used for finding Qutliers
def IQR(df,variable,distance):
    IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    upper = df[variable].quantile(0.75) + (IQR*distance)
    lower = df[variable].quantile(0.25) - (IQR*distance)
    return lower, upper
 
IQR(df, 'Months_Since_Deliquency',1.5)
#
df.Months_Since_Deliquency = df.Months_Since_Deliquency.map(lambda x : 95 if x > 95 else x)
sns.distplot(df.Months_Since_Deliquency)

df.Months_Since_Deliquency.fillna(0, inplace = True)
# Using cuberoot transformation
df.Months_Since_Deliquency = np.cbrt(df.Months_Since_Deliquency)

#Total_Accounts
sns.distplot(df.Total_Accounts)
df.Total_Accounts.describe()

IQR(df, 'Total_Accounts', 1.5)

df.Total_Accounts= df.Total_Accounts.map(lambda x : 58 if x > 58 else x)

df.Total_Accounts =np.sqrt(df.Total_Accounts)

# Here I used IQR function to find uper and lower bounds and bined with upper percentile.
# Further squareroot transform is used for decreasing skewness
#Number_Open_Accounts

sns.distplot(df.Number_Open_Accounts)

IQR(df,'Number_Open_Accounts', 1.5)
df.Number_Open_Accounts = df.Number_Open_Accounts.map(lambda x : 25 if x > 25 else x)
# after squareroot transformation

df.Number_Open_Accounts = np.sqrt(df.Number_Open_Accounts)
sns.distplot(df.Number_Open_Accounts)

# Closed account a new variable. Created by the difference of total and open accounts

df['closed_accounts']= df['Total_Accounts'] - df['Number_Open_Accounts']

IQR(df, 'closed_accounts', 1.5)


df.closed_accounts.describe()

sns.distplot(df.closed_accounts)

# gender 

df.Gender.replace({'Female' : 1, 'Male' : 0}, inplace = True)
# Annual_Income
# first removing outliers qith help of IQR

df.Annual_Income.describe().apply(lambda x: format(x, 'f'))
df.Annual_Income.median()
IQR(df,'Annual_Income', 1.5)

df.Annual_Income.describe()

df.Annual_Income = df.Annual_Income.map(lambda x :155000 if x >155000 else x)
# Filling null values with the help of correlated vriable total_accounts
df.Annual_Income = df.groupby('Total_Accounts')['Annual_Income'].transform(lambda x: x.fillna(x.mean()))
sns.distplot(df.Annual_Income)
# For skeweness
df.Annual_Income = np.cbrt(df.Annual_Income)
sns.distplot(df.Annual_Income)
# DRoping Gender column because of no relation
df = df.drop(['Loan_ID','Gender'], axis = 1)

training = df.iloc[:len(train),:]
testing = df.iloc[len(train):, : ]
testing = testing.drop('Interest_Rate', axis = 1)
# X,y variables

X = training.drop('Interest_Rate', axis = 1)
y = training['Interest_Rate']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score

x_train, x_val,y_train,y_val = train_test_split(X,y, test_size = 0.18, random_state = 21)


# The data is unbalanced
sns.countplot(df.Interest_Rate)
# These are the class weights for each class
class_1 =( len(y) - len(y[y==1]))/len(y)
class_2 =( len(y) - len(y[y==2]))/len(y)
class_3 =( len(y) - len(y[y==3]))/len(y)
print(class_1,class_2,class_3)

# its better to give a highter weight than the weight calculated. The idea is that model should know that class 2 is abundent.
weight = {1: '0.79', 2: '0.65', 3: '0.68'}

from lightgbm import LGBMClassifier
lgb =LGBMClassifier(boosting_type='gbdt',
                       max_depth=5,
                       learning_rate=0.05,
                       n_estimators=5000,
                       class_weight = weight,
                       min_child_weight = 0.02,
                       colsample_bytree=0.6, 
                       random_state=7,
                       objective='multiclass')

lgb.fit(x_train,y_train,
          eval_set=[(x_train,y_train),(x_val, y_val.values)],
          early_stopping_rounds=500,
          verbose=200)
# kfold
from sklearn.model_selection import StratifiedKFold

err = []
y_pred_tot_lgm = []

from sklearn.model_selection import StratifiedKFold

fold = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
i = 1
for train_index, test_index in fold.split(X, y):
    x_train, x_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = y.iloc[train_index], y.iloc[test_index]
    m = LGBMClassifier(boosting_type='gbdt',
                       max_depth=5,
                       learning_rate=0.05,
                       n_estimators=5000,
                       class_weight = weight,
                       min_child_weight = 0.02,
                       colsample_bytree=0.6, 
                       random_state=7,
                       objective='multiclass')
    m.fit(x_train, y_train,
          eval_set=[(x_train,y_train),(x_val, y_val)],
          early_stopping_rounds=200,
          verbose=200)
    pred_y = m.predict(x_val)
    print(i, " err_lgm: ", accuracy_score(y_val, pred_y))
    err.append(accuracy_score(y_val, pred_y))
    pred_test = m.predict(testing)
    i = i + 1
    y_pred_tot_lgm.append(pred_test)
    
print(np.mean(err,0))
err
err[12]

submission = pd.DataFrame()
submission['Loan_ID'] = test['Loan_ID']
submission['Interest_Rate'] = y_pred_tot_lgm[12]
submission.to_csv('normal_kfolds.csv', index=False, header=True)