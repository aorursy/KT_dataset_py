import pandas as pd

import numpy as np

import os

os.getcwd()
# traindt = pd.read_csv("Train.csv")

# testdt = pd.read_csv("Test.csv")



testdt = pd.read_csv("../input/Test.csv")

traindt = pd.read_csv("../input/Train.csv")
traindt.head(5)
testdt.isnull().sum()
def preprocessing_df(df):

    #drop loan_enq

    #df = df.drop(['loan_enq'],axis=1)

   

    #dropping rows with age>100

    #df=df.drop(df[df['age']>100].index) 

   

    return df
traindt = preprocessing_df(traindt)
testdt = preprocessing_df(testdt)
traindt.shape
testdt.shape
def detect_outliers(dataframe):

    cols = list(dataframe)

    outliers = pd.DataFrame(columns=['Feature','Number of Outliers'])

    

    for column in cols:

        if column in dataframe.select_dtypes(include=np.number).columns:

            q1 = dataframe[column].quantile(0.25) 

            q3 = dataframe[column].quantile(0.75)

            iqr = q3 - q1

            fence_low = q1 - (1.5*iqr)

            fence_high = q3 + (1.5*iqr)

            outliers = outliers.append({'Feature':column,'Number of Outliers':dataframe.loc[(dataframe[column] < fence_low) | (dataframe[column] > fence_high)].shape[0]},ignore_index=True)

    return outliers



detect_outliers(traindt)
X= traindt.drop(["ID","cc_cons"],axis=1)

y=traindt.cc_cons
X.head()

import seaborn as sns

import matplotlib.pyplot as plt
y = np.log1p(y)    #transfomed target variable because it was right skewed



sns.distplot(y)
#treat categorical data

# Split into categorical and numerical columns

X_cols = X.columns

num_cols = X.select_dtypes(exclude=['object','category']).columns

cat_cols = [i for i in X_cols if i not in X[num_cols].columns]

for i in cat_cols:

    X[i] = X[i].astype('category')
#X[num_cols]= np.log1p(X[num_cols])
X[num_cols]
cat_cols
num_cols
from scipy.stats.mstats import winsorize
 # Function to treat outliers 

def treat_outliers(dataframe):

    cols = list(dataframe)

    for col in cols:

        if col in dataframe.select_dtypes(include=np.number).columns:

            dataframe[col] = winsorize(dataframe[col], limits=[0.1, 0.1],inclusive=(True, True))

    

    return dataframe    



X[num_cols] = treat_outliers(X[num_cols])



# Checking for outliers after applying winsorization

detect_outliers(X)
num_cols
from sklearn.preprocessing import Binarizer 

X.head(5)
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler
#Label Encoding to be able to use categorical variables like age group in the regression eq

cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

le = LabelEncoder()

for i in cols:

    X[i] = le.fit_transform(X[i])




#X['gender'] = X['gender'].str.replace('M','1')

#X['gender'] = X['gender'].str.replace('F','0')

#X['gender'] = X['gender'].astype('int32')
X.account_type.value_counts()
#X['account_type'] = X['account_type'].str.replace('saving','1')

#X['account_type'] = X['account_type'].str.replace('current','0')

#X['account_type'] = X['account_type'].astype('int32')
X.head(5)
# Standardization

scaler = StandardScaler()

X = scaler.fit_transform(X)

X = pd.DataFrame(X, columns=X_cols)

X.cc_cons_apr.std()
sns.distplot(X.cc_cons_apr)
from sklearn.model_selection import train_test_split
# Spliting

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=72)
def rmsle(actual_column, predicted_column):

    assert len(actual_column) == len(predicted_column)

    sum=0.0

    for x,y in zip(actual_column, predicted_column):

        if x<0 or y<0: #check for negative values. 

            continue

        p = np.log(y+1)

        r = np.log(x+1)

        sum = sum + (p - r)**2

    return (sum/len(predicted_column))**0.5
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler

from sklearn.linear_model import LinearRegression,LogisticRegression

from sklearn.metrics import mean_squared_error,accuracy_score,mean_squared_log_error,r2_score
#X_1=X.drop(['investment_4'],axis=1)
y_train = np.exp(y_train)-1

y_train
y_test = np.exp(y_test)-1
y_test
y_train=np.log1p(y_train)

y_train
lr = LinearRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

y_pred_train = lr.predict(X_train)

print('Train RMSE:',np.sqrt(mean_squared_error(y_train, lr.predict(X_train))))

print('Test RMSE:',np.sqrt(mean_squared_error(y_test, lr.predict(X_test))))

rmsle(np.expm1(y_train), np.expm1(lr.predict(X_train)))
x_mean = np.mean(X_train)

x_mean
y_mean = np.mean(y_train)

y_mean
yy1 = np.expm1(y_train)

ypred1 = [np.mean(yy1) for i in range(yy1.shape[0])]

print(yy1.shape, len(ypred1))
rmsle(yy1,ypred1)
yy2 = np.expm1(y_train)

ypred2 = [np.mean(yy2) for i in range(y_test.shape[0])]

print(y_test.shape, len(ypred2))
rmsle(y_test,ypred2)
rmsle(y_test,y_pred)
lr.coef_
from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(random_state=42, min_samples_split=55, min_samples_leaf=120)

dt_reg.fit(X_train,y_train)



print('Train RMSE:',np.sqrt(mean_squared_error(y_train, dt_reg.predict(X_train))))

print('Test RMSE:',np.sqrt(mean_squared_error(y_test, dt_reg.predict(X_test))))

import sklearn

sklearn.tree.plot_tree(dt_reg)
y_pred_dt = dt_reg.predict(X_test)
rmsle(y_test,y_pred_dt)
testdt = pd.read_csv("Test.csv")

testdt= preprocessing_df(testdt)
testdt.isnull().sum()
testdt.head()

detect_outliers(testdt)
testdt[num_cols] = treat_outliers(testdt[num_cols])
detect_outliers(testdt)
testdt.head()
test_id_col = testdt['ID']
test_id_col.isnull().sum()
testdt = testdt.drop(['ID'],axis=1)
testdt.head()
# Label Encoding 

for i in cat_cols: 

    testdt[i] = le.fit_transform(testdt[i])
#testdt['gender'] = testdt['gender'].str.replace('M','1')

#testdt['gender'] = testdt['gender'].str.replace('F','0')

#testdt['gender'] = testdt['gender'].astype('int32')
testdt.account_type.value_counts()
#testdt['account_type'] = testdt['account_type'].str.replace('saving','1')

#testdt['account_type'] = testdt['account_type'].str.replace('current','0')

#testdt['account_type'] = testdt['account_type'].astype('int32')
testdt
X.shape
testdt.shape
X.columns
testdt.columns
testdt
testdt.columns
testdt_col = testdt.columns
testdt_col

# Standardization for Test

#scaler = StandardScaler()



testdt = scaler.transform(testdt)

testdt = pd.DataFrame(testdt, columns=testdt_col)

testdt
testdt.shape
testdt['cc_cons'] = lr.predict(testdt)
testdt['cc_cons'] = np.exp(testdt['cc_cons'])-1
submissions_lr = pd.concat([test_id_col, testdt['cc_cons']], axis=1) 

submissions_lr.to_csv('submission_LinearReg.csv', index=False) 
submissions_lr