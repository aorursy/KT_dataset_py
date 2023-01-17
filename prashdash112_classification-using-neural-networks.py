import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
df=pd.read_csv(r'../input/lending-club-loan-two-new-version/lending_club_loan_two.csv')
df.head(10)
df.info()
plt.figure(figsize=(12,6))
sns.heatmap(data=df.isnull())
df.columns
df=df.drop('address',axis=1)
df=df.drop('initial_list_status',axis=1)
df=df.drop('emp_title',axis=1)
plt.figure(figsize=(12,6))
sns.heatmap(data=df.isnull())
df[df['mort_acc'].isnull()==True]
plt.figure(figsize=(20,6))
sns.countplot(x='mort_acc',data=df)
df.corr().transpose()
df['mort_acc'].plot(kind='hist')
df['mort_acc'].fillna(0.0,inplace=True)
plt.figure(figsize=(20,6))
sns.countplot(x='mort_acc',data=df)
df['emp_length'].unique()
def num(a):
    if a=='10+ years':
        return 10
    elif a=='4 years':
        return 4
    elif a=='< 1 year':
        return 0.5
    elif a=='6 years':
        return 6
    elif a=='9 years':
        return 9
    elif a=='2 years':
        return 2
    elif a=='3 years':
        return 3
    elif a=='8 years':
        return 8
    elif a=='7 years':
        return 7
    elif a=='5 years':
        return 5
    elif a=='1 years':
        return 1
    
print(df[df['emp_length'].isnull()==True].shape)
print(df.shape)
print((18301/396030)*100)
#4.62% of data gets removed
df=df.dropna(axis='rows')
df
df['emp_length']=df['emp_length'].map(num)
df=df.drop('grade',axis=1)
df=df.drop('sub_grade',axis=1)
df
df.select_dtypes('object')
df['term'].unique()
df['term']=df['term'].map({' 36 months':36 , ' 60 months':60})
df=df.drop('earliest_cr_line',axis=1)
df=df.drop('issue_d',axis=1)
df=df.drop('verification_status',axis=1)
df=df.drop('title',axis=1)
df.select_dtypes('object')
from sklearn.preprocessing import OrdinalEncoder
o=OrdinalEncoder()
df[['home_ownership','loan_status','application_type']]=o.fit_transform(df[['home_ownership','loan_status','application_type']])
df=df.drop('purpose',axis=1)
df
plt.figure(figsize=(8,5))
sns.countplot(x='term',data=df)
plt.figure(figsize=(14,6))
sns.scatterplot(x='loan_amnt',y='int_rate',data=df,alpha=0.3)
plt.figure(figsize=(14,6))
sns.scatterplot(x='loan_amnt',y='installment',data=df,alpha=0.3,color='red')
plt.figure(figsize=(10,5))
sns.set(style="darkgrid")
sns.countplot(x='loan_status',data=df,hue='term')
plt.legend()
df=df.dropna()
X=df.drop('loan_status',axis=1).values
y=df['loan_status'].values
X.shape
y.shape
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40)
from sklearn.preprocessing import MinMaxScaler
scaled=MinMaxScaler()
scaled.fit(X_train)
X_train = scaled.transform(X_train)
X_test = scaled.transform(X_test)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation
model = Sequential()

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw


# input layer
model.add(Dense(16, activation='relu'))
# hidden layer
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.2))
# hidden layer
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.2))
# output layer
model.add(Dense(units=1,activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x=X_train, 
          y=y_train, 
          epochs=15,
          batch_size=128,
          validation_data=(X_test, y_test) 
          )
loss=pd.DataFrame(data=model.history.history)
loss.plot()
predictions = model.predict_classes(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))