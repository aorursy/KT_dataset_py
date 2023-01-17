import pandas as pd
data_info = pd.read_csv('../input/keras-data-sets/lending_club_info.csv',index_col='LoanStatNew')
print(data_info.loc['revol_util']['Description'])
def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])
feat_info('mort_acc')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# might be needed depending on your version of Jupyter
%matplotlib inline
df = pd.read_csv('../input/keras-data-sets/lending_club_loan_two.csv')
df.info()
# CODE HERE

sns.countplot(x='loan_status',data=df)

# CODE HERE
plt.figure(figsize=(12,6))
sns.distplot(df['loan_amnt'],kde=False,bins=60)

# CODE HERE
df.corr()

# CODE HERE
plt.figure(figsize=(12,7))
sns.heatmap(df.corr(),annot=True,cmap='plasma')
# CODE HERE
sns.scatterplot(x='installment',y='loan_amnt',data= df)
# CODE HERE
sns.boxplot(x='loan_status',y='loan_amnt',data=df)
# CODE HERE
df.groupby('loan_status')['loan_amnt'].describe()
# CODE HERE
df['grade'].unique()
df['sub_grade'].unique()
# CODE HERE
sns.countplot(x='grade',data=df,hue='loan_status')
#CODE HERE
plt.figure(figsize=(12,5))
sns.countplot(x='sub_grade',data=df,hue='loan_status',order=sorted(df['sub_grade'].unique()))
# CODE HERE
f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')
# CODE HERE
df['loan_repaid'] = df['loan_status'].apply(lambda x:1 if x=="Fully Paid" else 0 )
df['loan_repaid']
#CODE HERE
df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
df.isnull().sum()
# CODE HERE
df.shape
# CODE HERE
df.isnull().sum()
# CODE HERE
(df.isnull().sum()/len(df))*100
# CODE HERE
feat_info('emp_title')
feat_info('emp_length')
# CODE HERE
df['emp_title'].nunique()
df['emp_title'].value_counts()
# CODE HERE
df = df.drop('emp_title',axis=1)
df.columns
# CODE HERE
plt.figure(figsize=(12,5))
sns.countplot(x='emp_length',data=df,order=sorted(df['emp_length'].dropna().unique()))
# CODE HERE
plt.figure(figsize=(12,5))
sns.countplot(x='emp_length',data=df,hue='loan_status',order=sorted(df['emp_length'].dropna().unique()))
# CODE HERE
ChargedOff = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']
FullyPaid = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']
emp_len = ChargedOff/ FullyPaid
emp_len.plot(kind='bar')
# CODE HERE
df = df.drop('emp_length', axis=1)
df.columns
# CODE HERE
df['purpose'].head(10)

df['title'].head(10)
# CODE HERE
df = df.drop('title',axis=1)
df.columns
# CODE HERE
feat_info('mort_acc')
# CODE HERE
df['mort_acc'].value_counts()
df.corr()['mort_acc'].sort_values()
df.groupby('total_acc').mean()
total_acc_mean = df.groupby('total_acc').mean()['mort_acc']
total_acc_mean
# CODE HERE
df['mort_acc'] = df.apply(lambda x : total_acc_mean[x['total_acc']] if (np.isnan(x['mort_acc'])) else x['mort_acc'],axis=1)
df.isnull().sum()
# CODE HERE
df = df.dropna()
df.isnull().sum()
# CODE HERE
df.select_dtypes(['object']).columns
# CODE HERE
df['term'] = df['term'].apply(lambda term: int(term[:3]))
df
# CODE HERE
df =df.drop('grade',axis =1)
# CODE HERE
dummies = pd.get_dummies(df['sub_grade'],drop_first=True)
dummies
df = pd.concat([df.drop('sub_grade',axis=1),dummies],axis=1)
df.columns
# CODE HERE
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)
#CODE HERE
df['home_ownership'].value_counts()
#CODE HERE
df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)
df.columns
#CODE HERE
df['zip_code'] = df['address'].apply(lambda x:x.split(' ')[-1])
df['zip_code'].value_counts()
dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)
#CODE HERE
df = df.drop('issue_d',axis=1)
#CODE HERE
df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda x : x.split("-")[-1])
df = df.drop('earliest_cr_line',axis=1)
from sklearn.model_selection import train_test_split
# CODE HERE
df = df.drop('loan_status',axis=1)
df.columns
#CODE HERE
X= df.drop('loan_repaid',axis=1).values
y= df['loan_repaid'].values
sample = df.sample(frac=0.1,random_state=101)
print(len(df))
#CODE HERE
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
# CODE HERE
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
# CODE HERE
model = Sequential()

# Choose whatever number of layers/neurons you want.

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw

# Remember to compile()
model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=1,activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

# CODE HERE
model.fit(x=X_train, 
          y=y_train, 
          epochs=25,
          batch_size=256,
          validation_data=(X_test, y_test), 
          )

# CODE HERE
from tensorflow.keras.models import load_model
model.save('keras.h5')  


# CODE HERE
loss = pd.DataFrame(model.history.history)
loss[['loss','val_loss']].plot()

# CODE HERE
predictions = model.predict_classes(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer
new_customer.values
# CODE HERE
model.predict_classes(new_customer.values.reshape(1,78))
# CODE HERE
df.iloc[random_ind]['loan_repaid']