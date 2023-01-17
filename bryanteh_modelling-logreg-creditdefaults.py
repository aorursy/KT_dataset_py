# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os





# Any results you write to the current directory are saved as output.

train = pd.read_csv('/kaggle/input/home-credit-default-risk/application_train.csv')

test = pd.read_csv('/kaggle/input/home-credit-default-risk/application_test.csv')

print(train.shape)
train.head(6)

train.TARGET.value_counts()
train.dtypes.value_counts()
train.select_dtypes('object')
train.select_dtypes('object').nunique()
import matplotlib.pyplot as plt

import math

plt.hist(train.DAYS_EMPLOYED)

#train.DAYS_EMPLOYED.value_counts()

train.DAYS_EMPLOYED.replace({365243:np.nan},inplace=True)

train.DAYS_EMPLOYED.value_counts()

train['DAY_EMPLOYED_ANOM'] = train.apply(lambda x: 1 if math.isnan(x['DAYS_EMPLOYED']) else 0,axis = 1)

print(train['DAY_EMPLOYED_ANOM'].value_counts(),train.shape)
columns=list(train.columns)

i=1

plt.subplot(2,2,1)

plt.hist(list(train.iloc[:,i+1]))

plt.title(columns[i+1])

plt.subplot(2,2,2)

plt.hist(list(train.iloc[:,i+2]))

plt.title(columns[i+2])

plt.subplot(2,2,3)

plt.hist(list(train.iloc[:,i+3]))

plt.title(columns[i+3])

plt.subplot(2,2,4)

plt.hist(list(train.iloc[:,i+4]))

plt.title(columns[i+4])
#inspect missing values



def missing_values_table(df):

    mis_val=df.isnull().sum()    

    mis_val_perc=100*df.isnull().sum()/len(df)

    mis_val_table=pd.concat([mis_val, mis_val_perc], axis=1) 

    mis_val_table_ren_columns = mis_val_table.rename(columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    mis_val_table_ren_columns = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1] != 0].sort_values('% of Total Values', ascending=False).round(1)

    print ("Your selected data frame has " + str(df.shape[1]) + " columns.\n"+"There are " + str(mis_val_table_ren_columns.shape[0]) +

 " columns that have missing values.")

    return mis_val_table_ren_columns



miss = missing_values_table(train)

#print(miss.describe())

#print(miss[miss['% of Total Values']>45])

miss_col = list(miss[miss['% of Total Values']>45].index)

print(list(miss_col))

train = train.drop(columns=list(miss_col))

test = test.drop(columns=list(miss_col))
print(train.shape)

print(test.shape)
import seaborn as sns



train['CREDIT_INCOME_RATIO'] = train['AMT_CREDIT']/train['AMT_INCOME_TOTAL']

test['CREDIT_INCOME_RATIO'] = test['AMT_CREDIT']/test['AMT_INCOME_TOTAL']



train['ANNUITY_INCOME_RATIO'] = train['AMT_ANNUITY']/train['AMT_INCOME_TOTAL']

test['ANNUITY_INCOME_RATIO'] = test['AMT_ANNUITY']/test['AMT_INCOME_TOTAL']



train['GOODS_PRICE_INCOME_RATIO'] = train['AMT_GOODS_PRICE']/train['AMT_INCOME_TOTAL']

test['GOODS_PRICE_INCOME_RATIO'] = test['AMT_GOODS_PRICE']/test['AMT_INCOME_TOTAL']



#print(train['CREDIT_INCOME_RATIO'].describe(),train['CREDIT_ANNUITY_RATIO'].describe())

#mdf = pd.melt(train[['CREDIT_INCOME_RATIO','TARGET']], var_name=['TARGET'], value_name=['CREDIT_INCOME_RATIO'])

#print(mdf.head(5))

#sns.boxplot(x='TARGET',y='CREDIT_ANNUITY_RATIO',hue='TARGET', data=train)

#sns.boxplot(x="Trial", y="value", hue="Number", data=mdf)

print(train[train.TARGET==0].ANNUITY_INCOME_RATIO.describe(),train[train.TARGET==1].ANNUITY_INCOME_RATIO.describe())



train = train.drop(columns=['AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE'])

test = test.drop(columns=['AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE'])

print(train.shape)

print(test.shape)

print(train.columns)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le_count = 0



# Iterate through the columns

for col in train:

    if train[col].dtype == 'object':

        # If 2 or fewer unique categories

        if len(list(train[col].unique())) <= 2:

            # Train on the training data

            le.fit(train[col])

            # Transform both training and testing data

            train[col] = le.transform(train[col])

            test[col] = le.transform(test[col])

            

            # Keep track of how many columns were label encoded

            le_count += 1

            

print('%d columns were label encoded.' % le_count)



train = pd.get_dummies(train)

test = pd.get_dummies(test)

print('training shape: ',train.shape)

print('testing shape: ',test.shape)
train_label = train.TARGET

train, test = train.align(test,join='inner', axis = 1)

train['TARGET'] = train_label



print('training features shape:',train.shape)

print('testing features shape:',test.shape)
x_train= train.drop(columns=['TARGET'])

y_train = train['TARGET']

x_test = test

from sklearn.preprocessing import MinMaxScaler

from sklearn.impute import SimpleImputer



columns = list(x_train.columns)

columns_2 = list(x_test.columns)

imputer = SimpleImputer(missing_values=np.nan, strategy='median')

scaler = MinMaxScaler(feature_range=(0,1))



imputer.fit(x_train)

x_train = imputer.transform(x_train)

x_test = imputer.transform(x_test)



#Normalize based on Min_Max

#scaler.fit(x_train)

#x_train = scaler.transform(x_train)

#x_test = scaler.transform(x_test)



#Normalize based on Standard Deviation

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

print('training data shape',x_train)

print('testing data shape',x_test)

from sklearn.linear_model import LogisticRegression



model = LogisticRegression(C = 0.0001)

model.fit(x_train,y_train)

print('accuracy on training data', model.score(x_train,y_train))

y_predict = model.predict(x_test)

y_predict = model.predict_proba(x_test)

display(model.score)

print(pd.DataFrame(y_predict,columns=['no_difficulty','TARGET']))

y_predict = pd.DataFrame(y_predict,columns=['no_difficulty','TARGET'])

print(y_predict.describe())
#print(len(test['SK_ID_CURR']))

#print(len(pd.Series(y_predict)))

#print(test['SK_ID_CURR'].head(10))

submission = pd.concat([test['SK_ID_CURR'],y_predict.TARGET], axis=1)

sns.boxplot(y='TARGET',data=submission)

print(submission.TARGET.describe())



submission.TARGET = submission.TARGET.apply(lambda x: 1 if x > 0.1 else 0)

print(submission.head(30))

print(submission.TARGET.value_counts())

submission.to_csv('baseline.csv',index=False)