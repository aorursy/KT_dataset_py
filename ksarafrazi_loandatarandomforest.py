import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns

%matplotlib inline 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv('../input/loan_borowwer_data.csv')
df.info()
sns.distplot(df['fico'][df['credit.policy']==0],kde=False,bins=25)
sns.distplot(df['fico'][df['credit.policy']==1],kde=False,color='r',bins=25)
sns.set(rc={'figure.figsize':(10,5)})
sns.countplot(df['purpose'],hue=df['not.fully.paid'])
sns.jointplot('fico','int.rate',df)
sns.lmplot('fico','int.rate',df,hue ='credit.policy',col='not.fully.paid' )
df.head()
Policy = pd.get_dummies(df['credit.policy'],drop_first=True)
Inq = pd.get_dummies(df['inq.last.6mths'],drop_first=True)
Delinq = pd.get_dummies(df['delinq.2yrs'],drop_first=True)
Pub = pd.get_dummies(df['pub.rec'],drop_first=True)
Purpose = pd.get_dummies(df['purpose'],drop_first=True)

df.drop(['credit.policy','inq.last.6mths','delinq.2yrs','pub.rec','purpose'],axis=1,inplace=True)
df = pd.concat([df,Policy,Inq,Delinq,Pub,Purpose] , axis =1)

#Decision Tree Classifier
x_train, x_test, y_train, y_test = train_test_split(df.drop('not.fully.paid',axis=1), 
                                                    df['not.fully.paid'], test_size=0.4)

dtreec = DecisionTreeClassifier()
dtreec.fit(x_train,y_train)
predict = dtreec.predict(x_test)

print("The confusion Matrix is : \n",confusion_matrix(y_test,predict))
print('\n')
print("The classification report  : \n",classification_report(y_test,predict))

#Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=300) 
rfc.fit(x_train,y_train)
rfc_pred = rfc.predict(x_test)

print("The confusion Matrix is : \n",confusion_matrix(y_test,rfc_pred))
print('\n')
print("The classification report  : \n",classification_report(y_test,rfc_pred))
