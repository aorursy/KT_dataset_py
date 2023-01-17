# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/lending-club-loan-two-new-version/lending_club_loan_two.csv')

df
df.info()
sns.countplot(x='loan_status',data=df)
plt.figure(figsize=(12,4))

sns.distplot(df['loan_amnt'],kde=False,bins=40)

plt.xlim(0,45000)
df.corr()
plt.figure(figsize=(12,7))

sns.heatmap(df.corr(),annot=True,cmap='viridis')

plt.ylim(10, 0)
plt.figure(figsize=(14,6))

sns.scatterplot(x='installment',y='loan_amnt',data=df,)
sns.boxplot(x='loan_status',y='loan_amnt',data=df)
df.groupby('loan_status')['loan_amnt'].describe()
sorted(df['grade'].unique())
sorted(df['sub_grade'].unique())
sns.countplot(x='grade',data=df,hue='loan_status')
plt.figure(figsize=(12,4))

subgrade_order = sorted(df['sub_grade'].unique())

sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' )
plt.figure(figsize=(12,4))

subgrade_order = sorted(df['sub_grade'].unique())

sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' ,hue='loan_status')
f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]



plt.figure(figsize=(12,4))

subgrade_order = sorted(f_and_g['sub_grade'].unique())

sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')
df['loan_status'].unique()
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
df[['loan_repaid','loan_status']]
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
df.head()
len(df)
df.isnull().sum()
100* df.isnull().sum()/len(df)
df['emp_title'].nunique()
df['emp_title'].value_counts()
df = df.drop('emp_title',axis=1)
sorted(df['emp_length'].dropna().unique())
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



sns.countplot(x='emp_length',data=df,order=emp_length_order)
plt.figure(figsize=(12,4))

sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')
emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']
emp_len = emp_co/emp_fp
emp_len
emp_len.plot(kind='bar')
df = df.drop('emp_length',axis=1)
df.isnull().sum()
df['purpose'].head(10)
df['title'].head(10)
df = df.drop('title',axis=1)
df['mort_acc'].value_counts()
print("Correlation with the mort_acc column")

df.corr()['mort_acc'].sort_values()
print("Mean of mort_acc column per total_acc")

df.groupby('total_acc').mean()['mort_acc']
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
total_acc_avg[2.0]
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
df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)
df.isnull().sum()
df = df.dropna()
df.isnull().sum()
df.select_dtypes(['object']).columns
df['term'].value_counts()
# Or just use .map()

df['term'] = df['term'].apply(lambda term: int(term[:3]))
df = df.drop('grade',axis=1)
subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)
df.columns
df.select_dtypes(['object']).columns
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)

df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)

df = pd.concat([df,dummies],axis=1)
df['home_ownership'].value_counts()
df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')



dummies = pd.get_dummies(df['home_ownership'],drop_first=True)

df = df.drop('home_ownership',axis=1)

df = pd.concat([df,dummies],axis=1)
df['zip_code'] = df['address'].apply(lambda address:address[-5:])
dummies = pd.get_dummies(df['zip_code'],drop_first=True)

df = df.drop(['zip_code','address'],axis=1)

df = pd.concat([df,dummies],axis=1)
df = df.drop('issue_d',axis=1)
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))

df = df.drop('earliest_cr_line',axis=1)
df.select_dtypes(['object']).columns
from sklearn.model_selection import train_test_split
df = df.drop('loan_status',axis=1)
X = df.drop('loan_repaid',axis=1).values

y = df['loan_repaid'].values
# df = df.sample(frac=0.1,random_state=101)

print(len(df))
#CODE HERE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
# CODE HERE
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

model.add(Dense(units=1,activation='sigmoid'))



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

random_ind = random.randint(0,len(df))



new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]

new_customer
model.predict_classes(new_customer.values.reshape(1,78))
df.iloc[random_ind]['loan_repaid']