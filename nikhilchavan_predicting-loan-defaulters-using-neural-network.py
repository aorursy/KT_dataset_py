import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
data_info = pd.read_csv('../input/lending-club-data/lending_club_loan_two.csv')
data_info.head()
data_info.isnull().sum()
sns.countplot(x='loan_status',data=data_info)
plt.figure(figsize = (14,8))

sns.distplot(data_info['loan_amnt'],kde = False)
data_info.corr()
plt.figure(figsize=(12,7))

sns.heatmap(data_info.corr(),annot=True,cmap='viridis')

plt.ylim(10, 0)
plt.figure(figsize=(12,7))

sns.scatterplot(x='loan_amnt',y='installment',data=data_info)
sns.boxplot(x='loan_status',y='loan_amnt',data=data_info)
data_info.groupby('loan_status')['loan_amnt'].describe()
sorted(data_info['grade'].unique())
sorted(data_info['sub_grade'].unique())
sns.countplot(x='grade',data=data_info,hue='loan_status')
plt.figure(figsize=(12,4))

subgrade_order = sorted(data_info['sub_grade'].unique())

sns.countplot(x='sub_grade',data=data_info,order = subgrade_order,palette='coolwarm' )
plt.figure(figsize=(12,4))

subgrade_order = sorted(data_info['sub_grade'].unique())

sns.countplot(x='sub_grade',hue = 'loan_status',data=data_info,order = subgrade_order,palette='coolwarm' )
f_and_g = data_info[(data_info['grade']=='G') | (data_info['grade']=='F')]



plt.figure(figsize=(12,4))

subgrade_order = sorted(f_and_g['sub_grade'].unique())

sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')
data_info['loan_repaid'] = data_info['loan_status'].map({'Fully Paid':1,'Charged Off':0})
data_info['loan_repaid'].unique()
data_info[['loan_repaid','loan_status']]
data_info.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
len(data_info)
data_info.isnull().sum()
data_info.isnull().sum()/len(data_info)
data_info['emp_title'].nunique()
data_info['emp_title'].value_counts()
data_info = data_info.drop('emp_title',axis=1)
sorted(data_info['emp_length'].dropna().unique())
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



sns.countplot(x='emp_length',data=data_info,order=emp_length_order)
plt.figure(figsize=(12,4))

sns.countplot(x='emp_length',data=data_info,order=emp_length_order,hue='loan_status')
emp_co = data_info[data_info['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']

emp_fp = data_info[data_info['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']

emp_len = emp_co/emp_fp

emp_len
emp_len.plot(kind='bar')
data_info = data_info.drop('emp_length',axis=1)
data_info.isnull().sum()
data_info['purpose'].head(10)
data_info['title'].head(10)
data_info['title'].head(10)
data_info['mort_acc'].value_counts()
data_info.corr()['mort_acc'].sort_values()
print("Mean of mort_acc column per total_acc")

data_info.groupby('total_acc').mean()['mort_acc']
total_acc_avg = data_info.groupby('total_acc').mean()['mort_acc']
total_acc_avg
total_acc_avg[2.0]
total_acc_avg
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
data_info['mort_acc'] = data_info.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
data_info.isnull().sum()
data_info.drop('title',axis = 1,inplace = True)
data_info.isnull().sum()
data_info = data_info.dropna()
data_info.isnull().sum()
data_info.select_dtypes(['object']).columns
data_info['term'].value_counts()
data_info['term'] = data_info['term'].apply(lambda term: int(term[:3])) 

# to get numeric field for term
data_info['term'].head()
data_info = data_info.drop('grade',axis = 1)
subgrade_dummies = pd.get_dummies(data_info['sub_grade'],drop_first=True)
data_info = pd.concat([data_info.drop('sub_grade',axis=1),subgrade_dummies],axis=1)
data_info.columns
data_info.select_dtypes(['object']).columns
dummies = pd.get_dummies(data_info[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)

data_info = data_info.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)

data_info = pd.concat([data_info,dummies],axis=1)
data_info['home_ownership'].value_counts()
data_info['home_ownership']=data_info['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')



dummies = pd.get_dummies(data_info['home_ownership'],drop_first=True)

data_info = data_info.drop('home_ownership',axis=1)

data_info = pd.concat([data_info,dummies],axis=1)
data_info['address']
data_info['zip_code'] = data_info['address'].apply(lambda address:address[-5:]) # extract zipcode
data_info['zip_code'].head()
data_info['zip_code'].nunique()
dummies = pd.get_dummies(data_info['zip_code'],drop_first=True)

data_info = data_info.drop(['zip_code','address'],axis=1)

data_info = pd.concat([data_info,dummies],axis=1)
data_info.head()
#This would be data leakage, we wouldn't know beforehand whether or not a loan would be issued when using our model, so in theory we wouldn't have an issue_date, drop this feature.



data_info = data_info.drop('issue_d',axis=1)
#This appears to be a historical time stamp feature. Extract the year from this feature using a .apply function, then convert it to a numeric feature. Set this new data to a feature column called 'earliest_cr_year'.Then drop the earliest_cr_line feature.

data_info['earliest_cr_year'] = data_info['earliest_cr_line'].apply(lambda date:int(date[-4:]))

data_info = data_info.drop('earliest_cr_line',axis=1)
data_info.select_dtypes(['object']).columns
from sklearn.model_selection import train_test_split
data_info = data_info.drop('loan_status',axis=1)
X = data_info.drop('loan_repaid',axis=1).values

y = data_info['loan_repaid'].values
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



# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw





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

model.add(Dense(units=1,activation='sigmoid')) #since it is binary a classification problem



# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train, 

          y=y_train, 

          epochs=25,

          batch_size=256,

          validation_data=(X_test, y_test), 

          )
from tensorflow.keras.models import load_model
model.save('full_data_project_model.h5')  
losses = pd.DataFrame(model.history.history)
losses[['loss','val_loss']].plot()
from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))
confusion_matrix(y_test,predictions)
import random

random.seed(101)

random_ind = random.randint(0,len(data_info))



new_customer = data_info.drop('loan_repaid',axis=1).iloc[random_ind]

new_customer
model.predict_classes(new_customer.values.reshape(1,78))
data_info.iloc[random_ind]['loan_repaid']