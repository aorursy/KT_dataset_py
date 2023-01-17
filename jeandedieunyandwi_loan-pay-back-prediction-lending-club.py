

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

        

loan_data = pd.read_csv('/kaggle/input/lending-club-dataset/lending_club_loan_two.csv')

loan_data.info()
sns.countplot(x='loan_status',data=loan_data)
loan_data['loan_status'].value_counts()
plt.figure(figsize=(12,4))

sns.distplot(loan_data['loan_amnt'],kde=False,bins=40)

plt.xlim(0,45000)
#How features collerate to one another

loan_data.corr()
plt.figure(figsize=(12,7))

sns.heatmap(loan_data.corr(),annot=True,cmap='viridis')

plt.ylim(10, 0)
sns.scatterplot(x='installment',y='loan_amnt',data=loan_data)
sns.boxplot(x='loan_status',y='loan_amnt',data=loan_data)
loan_data.groupby('loan_status')['loan_amnt'].describe()
sorted(loan_data['grade'].unique())
sorted(loan_data['sub_grade'].unique())
sns.countplot(x='grade',data=loan_data,hue='loan_status')
plt.figure(figsize=(12,4))

subgrade_order = sorted(loan_data['sub_grade'].unique())

sns.countplot(x='sub_grade',data=loan_data,order = subgrade_order,palette='coolwarm' )
plt.figure(figsize=(12,4))

subgrade_order = sorted(loan_data['sub_grade'].unique())

sns.countplot(x='sub_grade',data=loan_data,order = subgrade_order,palette='coolwarm' ,hue='loan_status')
f_and_g = loan_data[(loan_data['grade']=='G') | (loan_data['grade']=='F')]



plt.figure(figsize=(12,4))

subgrade_order = sorted(f_and_g['sub_grade'].unique())

sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')
loan_data['loan_status'].unique()
loan_data['loan_repaid'] = loan_data['loan_status'].map({'Fully Paid':1,'Charged Off':0})
loan_data[['loan_repaid','loan_status']]
#Let's start by showing sum of missing data

loan_data.isnull().sum()
#By percentage of dataframe, missing data are

100* loan_data.isnull().sum()/len(loan_data)
loan_data['emp_title'].nunique()
loan_data['emp_title'].value_counts()
loan_data=loan_data.drop('emp_title', axis=1)
#Let's create the count plot of the emp_length column by sorting firdt the order of the values

sorted(loan_data['emp_length'].dropna().unique())
emp_length_order = [ '< 1 year',

                      '1 year',

                     '2 years',

                     '3 years',

                     '4 years',

                     '5 years',

                     '6 years',

                     '7 years',

                     '8 years',

                     '9 years',

                     '10+ years']
plt.figure(figsize=(12,4))



sns.countplot(x='emp_length',data=loan_data,order=emp_length_order)
plt.figure(figsize=(12,4))

sns.countplot(x='emp_length',data=loan_data,order=emp_length_order,hue='loan_status')
emp_co = loan_data[loan_data['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_fp = loan_data[loan_data['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']
emp_len = emp_co/emp_fp

emp_len
emp_len.plot(kind='bar')
loan_data=loan_data.drop('emp_length',axis=1)
loan_data.isnull().sum()
loan_data['purpose'].head(10)
loan_data['title'].head(10)
loan_data=loan_data.drop('title',axis=1)
loan_data['mort_acc'].value_counts()
print("Correlation with the mort_acc column")

loan_data.corr()['mort_acc'].sort_values()
print("Mean of mort_acc column per total_acc")

loan_data.groupby('total_acc').mean()['mort_acc']
total_acc_avg =loan_data.groupby('total_acc').mean()['mort_acc']
def fill_mort_acc(total_acc,mort_acc):

    '''

    Accepts the total_acc and mort_acc values for the row.

    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value

    for the corresponding total_acc value for that row.

    

    total_acc_avg here should be a Series or dictionary containing the mapping of the

    groupby averages of mort_acc per total_acc values.

    '''

    if np.isnan(mort_acc):

        return total_acc_avg[total_acc]

    else:

        return mort_acc
loan_data['mort_acc'] = loan_data.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
loan_data.isnull().sum()
loan_data=loan_data.dropna()
loan_data.isnull().sum()
print('The categorical variables are:')

loan_data.select_dtypes(['object']).columns
loan_data['term'].value_counts()
loan_data['term'] = loan_data['term'].apply(lambda term: int(term[:3]))
loan_data=loan_data.drop('grade',axis=1)
subgrade_dummies = pd.get_dummies(loan_data['sub_grade'],drop_first=True)
loan_data = pd.concat([loan_data.drop('sub_grade',axis=1),subgrade_dummies],axis=1)
loan_data.columns
loan_data.select_dtypes(['object']).columns
dummies = pd.get_dummies(loan_data[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)

loan_data = loan_data.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)

loan_data = pd.concat([loan_data,dummies],axis=1)
loan_data['home_ownership'].value_counts()
loan_data['home_ownership']=loan_data['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')



dummies = pd.get_dummies(loan_data['home_ownership'],drop_first=True)

loan_data = loan_data.drop('home_ownership',axis=1)

loan_data = pd.concat([loan_data,dummies],axis=1)
loan_data['zip_code'] = loan_data['address'].apply(lambda address:address[-5:])
dummies = pd.get_dummies(loan_data['zip_code'],drop_first=True)

loan_data = loan_data.drop(['zip_code','address'],axis=1)

loan_data = pd.concat([loan_data,dummies],axis=1)
loan_data=loan_data.drop('issue_d',axis=1)
loan_data['earliest_cr_year'] = loan_data['earliest_cr_line'].apply(lambda date:int(date[-4:]))

loan_data = loan_data.drop('earliest_cr_line',axis=1)
loan_data.select_dtypes(['object']).columns
loan_data = loan_data.drop('loan_status',axis=1)
from sklearn.model_selection import train_test_split
X= loan_data.drop('loan_repaid',axis=1).values

y = loan_data['loan_repaid'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Activation,Dropout

from tensorflow.keras.constraints import max_norm
model = Sequential()



# input layer

model.add(Dense(78,  activation='relu'))

model.add(Dropout(0.2))



# hidden layer

model.add(Dense(39, activation='relu'))

model.add(Dropout(0.2))



# hidden layer

model.add(Dense(19, activation='relu'))

model.add(Dropout(0.2))



# output layer

model.add(Dense(units=1,activation='sigmoid'))



# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train, 

          y=y_train, 

          epochs=25,

          batch_size=256,

          validation_data=(X_test, y_test), 

          )
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))
confusion_matrix(y_test,predictions)
import random

random.seed(101) ##Keeping same random numbers 

random_ind = random.randint(0,len(loan_data))



new_customer = loan_data.drop('loan_repaid',axis=1).iloc[random_ind]

new_customer
model.predict_classes(new_customer.values.reshape(1,78))