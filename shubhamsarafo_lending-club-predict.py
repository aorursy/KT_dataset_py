import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

### data information setting

data_info = pd.read_csv('/kaggle/input/lending-club/lending_club_info.csv',index_col='LoanStatNew')
print(data_info.loc['revol_util']['Description'])
print(data_info)
def feat_info(col_name):

    print(data_info.loc[col_name]['Description'])
feat_info('annual_inc')
df = pd.read_csv('/kaggle/input/lending-club/lending_club_loan_two.csv')
df.head()
df.info()
sns.countplot(x='loan_status',data=df)
plt.figure(figsize=(12,4))

sns.distplot(df['loan_amnt'],kde=False)
feat_info('loan_amnt')
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),annot=True,cmap="YlGnBu")
feat_info('installment')
feat_info('loan_amnt')
sns.scatterplot(x='installment',y='loan_amnt',data=df)
sns.boxplot(x='loan_status',y='loan_amnt',data=df)
df.groupby('loan_status')['loan_amnt'].describe()
df.grade.unique()
df.sub_grade.unique()
feat_info('grade')
feat_info('sub_grade')
plt.figure(figsize=(10,4))

grade_order = sorted(df['grade'].unique())

sns.countplot(x='grade',hue='loan_status',data=df,order=grade_order)
plt.figure(figsize=(12,4))

subgrade_order = sorted(df['sub_grade'].unique())

sns.countplot(x='sub_grade',data=df,order=subgrade_order,palette='coolwarm')
plt.figure(figsize=(12,4))

sns.countplot(x = 'sub_grade' ,data = df , order = subgrade_order , palette = 'coolwarm' ,hue='loan_status')
f_and_g = df[(df['grade'] == 'G') | (df['grade'] == 'F') ]



plt.figure(figsize=(12,4))

subgrade_order = sorted(f_and_g['sub_grade'].unique())

sns.countplot(x='sub_grade',data=df ,order = subgrade_order ,palette='coolwarm' ,hue='loan_status')
df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})
df[['loan_repaid','loan_status']].head()
plt.figure(figsize=(8,3))

df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')
df.columns
feat_info('revol_util')
df.head()
len(df)
df.isnull().sum()
100 * df.isnull().sum() / len(df)
feat_info('mort_acc')

print('\n')

feat_info('emp_title')

print('\n')

feat_info('emp_length')
df['emp_title'].nunique()
# df['emp_title'].value_counts()

# there are way too many unique values to convert to dummy
df = df.drop('emp_title',axis=1)
sorted(df['emp_length'].dropna().unique())
emp_length_order = ['< 1 year',

 '1 year',

 '2 years',

 '3 years',

 '4 years',

 '5 years',

 '6 years',

 '7 years',

 '8 years',

 '9 years',

 '10+ years'                   

 ]
plt.figure(figsize=(10,4))

sns.countplot(x='emp_length',data=df,order= emp_length_order,palette='viridis',hue='loan_status')

plt.tight_layout()
emp_co = df[df['loan_status'] == 'Charged Off'].groupby('emp_length').count()['loan_status']
emp_fp = df[df['loan_status'] == 'Fully Paid'].groupby('emp_length').count()['loan_status']
emp_co / emp_fp
# no major difference here too

# dropping emp_length column 
df = df.drop('emp_length',axis=1)
df.isnull().sum()
feat_info('title')
df['title'].head()
df = df.drop('title',axis=1)
feat_info('mort_acc')
df['mort_acc'].value_counts()
df.corr()['mort_acc'].sort_values()
# we will group the dataframe by the total_acc and calculate the mean value for the 

# mort_acc per total_acc entry
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']
# if the mort_acc is missing , then we will fill in the mean value coressponding to its total_acc_value from the

# series we created above.
def fill_mort_aacc( total_acc, mort_acc):

    

    if np.isnan(mort_acc):

        return total_acc_avg[total_acc]

    else:

        return mort_acc
df['mort_acc'] = df.apply(lambda x:fill_mort_aacc(x['total_acc'],x['mort_acc']),axis=1)
df.isnull().sum()
# we will just drop the rows for revol_util , pub_rec_bankruptcies
df = df.dropna()
df.isnull().sum()
df
df.select_dtypes(['object']).columns
#let us go through each of these
feat_info('term')
df['term'].value_counts()
df['term'] = df['term'].apply(lambda term: int(term[:3]))
df.term.head()
# we already know that it has a subgrade feature
df = df.drop('grade',axis=1)
dummies = pd.get_dummies(df['sub_grade'], drop_first=True)



df = pd.concat([df.drop('sub_grade',axis=1),dummies],axis=1)
# df.columns
dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)

df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)

df = pd.concat([df,dummies],axis=1)
feat_info('home_ownership')
df['home_ownership'].value_counts()
#let us put 'none' and 'any' in 'OTHER' column
df['home_ownership'] = df['home_ownership'].replace(['NONE','ANY'],'OTHER')
dummies = pd.get_dummies(df['home_ownership'],drop_first=True)



df = pd.concat([df.drop('home_ownership',axis=1),dummies],axis=1)
#lets get the zipcode out of the address
df['zipcode'] = df['address'].apply(lambda x:x[-5:])
df.zipcode.value_counts()
dummies = pd.get_dummies(df['zipcode'],drop_first=True)



df = pd.concat([df.drop('zipcode',axis=1),dummies],axis=1)
df = df.drop('address',axis=1)
feat_info('issue_d')
df = df.drop('issue_d',axis=1)
feat_info('earliest_cr_line')
df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))
# df['earliest_cr_line'].value_counts()
from sklearn.model_selection import train_test_split
#dropping the loan_status column since it is a duplicate of the loan_repaid column
df = df.drop('loan_status', axis = 1)
## setting up X and y
X = df.drop('loan_repaid',axis=1).values

y = df['loan_repaid'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
#prevent data leakage

X_test = scaler.transform(X_test)
import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout
# 78 --> 39 --> 19 --> 1 
X_train.shape

#since there are 78 feature .. we add first layer as 78
model = Sequential()



model.add(Dense(78,activation = 'relu'))

model.add(Dropout(0.2))



model.add(Dense(39,activation = 'relu'))

model.add(Dropout(0.2))



model.add(Dense(19,activation = 'relu'))

model.add(Dropout(0.2))



model.add(Dense(units=1,activation='sigmoid'))



model.compile(loss='binary_crossentropy',optimizer= 'adam')
model.fit(x=X_train, 

          y=y_train, 

          epochs=25,

          batch_size=256,

          validation_data=(X_test, y_test), 

          )
losses = pd.DataFrame(model.history.history)
losses.plot(figsize=(10,4))
from sklearn.metrics import classification_report,confusion_matrix
predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))
df['loan_repaid'].value_counts()
# the below code shows the minimum thresh-hold the model accuracy always should be.

# because loan repaid in the original dataset is already inclined to what the loan should be already 
317696 / len(df)
# A straight guess will give 80% accuracy

# A random guess will give 50%

# our model is giving 89% accuracy
print(confusion_matrix(y_test,predictions))
import random

random.seed(101)

random_ind = random.randint(0,len(df))



new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]

new_customer
new_customer = scaler.transform(new_customer.values.reshape(1,78))
model.predict_classes(new_customer)
# df.drop('loan_repaid',axis=1).iloc[random_ind]

df.iloc[random_ind]['loan_repaid']