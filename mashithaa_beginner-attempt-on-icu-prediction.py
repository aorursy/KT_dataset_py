import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# reading the csv
df=pd.read_csv('../input/covid19-patient-precondition-dataset/covid.csv')
df.head()
df['icu'].unique
#mapping 97 & 98 categories to category 3 whic is NA
df['icu']=df['icu'].map({1:1,2:2,97:3,99:3})
df.head()
sns.countplot(x='icu',data=df)
#dropping the icu values where value == 3 which are NA's 
indexnames=df[df['icu']==3].index
df.drop(indexnames,inplace=True)
# now check the icu values on count plot ,only 1 and 2 cateogries are present
sns.countplot(x='icu',data=df)
df['pregnancy']
# mapping the 97and 98 cateogories to NA 
df['pregnancy']=df['pregnancy'].map({1:1,2:2,97:3,98:3})
df['pregnancy'].value_counts()
df.head()
df['date_died'].value_counts()
#drop id ,patient_type column which is not required
df.drop(['id','patient_type'],axis=1,inplace=True)
df.head()
sns.countplot(x='icu',data=df,hue='sex')
sns.distplot(df['age'],kde=False,bins=20)
#drop dates column which is not required
df.drop(['entry_date','date_symptoms','date_died'],axis=1,inplace=True)
df.head()
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(),annot=True,cmap='viridis')
df.corr()['icu'][:-1].sort_values().plot(kind='bar')
X = df.drop('icu',axis=1)
y = df['icu']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)
#scaling the features set
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train= scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train.shape
X_test.shape
#using random forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))
#using decision tree
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predictions = dtree.predict(X_test)

print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
#random forest is gving the better accuracy than decision tree
# brand new prediction 
single_person = df.drop('icu',axis=1).iloc[3]
single_person
single_person = scaler.transform(single_person.values.reshape(-1, 17))
single_person
rfc.predict(single_person)
df.iloc[3]
