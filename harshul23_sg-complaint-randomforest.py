# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas
df=pd.read_csv('../input/train.csv')
df.head()
df[['Complaint-ID','Complaint-Status']].groupby('Complaint-Status').count()
df.describe()
date1=pd.to_datetime(df["Date-received"])
date2=pd.to_datetime(df["Date-sent-to-company"])
y=df['Complaint-Status']
x=df.drop(columns=['Complaint-ID','Date-received','Date-sent-to-company','Consumer-complaint-summary','Complaint-Status'])
Y=pd.get_dummies(y)
X=pd.get_dummies(x)
X['days between received and sent'] = (date2-date1).dt.days

X.head()
X=X.drop(columns=["Complaint-reason_Account terms and changes","Complaint-reason_Advertising",
                  "Complaint-reason_Incorrect exchange rate",
                  "Complaint-reason_Problem with an overdraft",
                  "Complaint-reason_Was approved for a loan, but didn't receive the money"])
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
#params=[{'n_estimators':[300,500,700], 'criterion':['gini','entropy'], 'min_samples_split':[2,10,20] }]
#grid=GridSearchCV(estimator=RandomForestClassifier(),param_grid=params,scoring='accuracy',n_jobs=-1)
clf = RandomForestClassifier(n_estimators=300,criterion='entropy',min_samples_split=30)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.15)
X_train['days between received and sent']=X_train['days between received and sent']/max(X_train['days between received and sent'])
X_test['days between received and sent']=X_test['days between received and sent']/max(X_test['days between received and sent'])

#grids=grid.fit(X_train,y_train)
#best_acc=grids.best_score_
#best_param=grids.best_params_
#print(best_acc,best_param)

clf.fit(X_train, y_train)

y_pred =  clf.predict(X_test)

from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
print('Accuracy Score on train data: ', accuracy_score(y_true=y_train, y_pred=clf.predict(X_train)))
print('Accuracy Score on test data: ', accuracy_score(y_true=y_test, y_pred=y_pred))

d=pd.read_csv('../input/sample_submission.csv')
d.head()
df2=pd.read_csv('../input/test.csv')
df2.head()
date12=pd.to_datetime(df2["Date-received"])
date22=pd.to_datetime(df2["Date-sent-to-company"])
idn=df2["Complaint-ID"]

x2=df2.drop(columns=['Complaint-ID','Date-received','Date-sent-to-company','Consumer-complaint-summary'])
X2=pd.get_dummies(x2)
X2=X2.drop(columns=["Complaint-reason_Can't stop withdrawals from your bank account"])
y_pred2 =  clf.predict(X2)

y_pred2[0:5]
l=[]
for y in y_pred2:
    if y[0]==1:
        l.append('Closed')
    if y[1]==1:
        l.append('Closed with explanation')
    if y[2]==1:
        l.append('Closed with monetary relief')
    if y[3]==1:
        l.append('Closed with non-monetary relief')
    if y[4]==1:
        l.append('Untimely response')
    
out = pd.DataFrame({'Closed':y_pred2[:,0],'Closed with explanation':y_pred2[:,1]
                   ,'Closed with monetary relief':y_pred2[:,2],'Closed with non-monetary relief':y_pred2[:,3]
                   ,'Untimely response':y_pred2[:,4]})
out = out.idxmax(axis=1)
out.columns = ['Complaint-Status']
out.name="Complaint-Status"
result = pd.concat([idn, out], axis=1)
result.head()
result.to_csv("result.csv",index = False)
