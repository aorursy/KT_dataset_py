import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.svm import SVC
data=pd.read_csv('../input/red-wine-quality-cortez-et-al-2009/winequality-red.csv')
data.head()
data.shape
data.isnull().sum()
data.describe()
data['quality'].unique()
data.info()
sns.countplot(data['quality'])
data.corr()
data_columns=data.columns
for ele in data_columns:

    fig = plt.figure(figsize = (10,6))

    sns.barplot(x = 'quality', y = ele, data = data)
for i in range(1599):

    if(data['quality'][i]<=6.5):

        data['quality'][i]=0

    else:

        data['quality'][i]=1
data['quality'].unique()
sns.countplot(data['quality'])
model=LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=1000)
y=data['quality']

X=data.drop(['quality'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
model.fit(X_train,y_train)
model.score(X_test,y_test)
rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))
clf = SVC() 
clf.fit(X_train, y_train) 
pred_svc=clf.predict(X_test)
print(classification_report(y_test, pred_svc))
print(confusion_matrix(y_test, pred_svc))
data_1=data[data.quality == 1]
data_0=data[data.quality == 0]
data_1.info()
data_0.info()
data_1_new = pd.concat([data_1, data_1],ignore_index=True, sort =False)
data_1_new.info()
data_new = pd.concat([data_1_new,data_0],ignore_index=True, sort =False)
data_new.info()
y_new=data_new['quality']

X_new=data_new.drop(['quality'],axis=1)
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.20)
model=LogisticRegression(solver='lbfgs',multi_class='auto',max_iter=1000)
model.fit(X_train,y_train)
model.score(X_test,y_test)
rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)
print(classification_report(y_test, pred_rfc))
clf = SVC() 
clf.fit(X_train, y_train) 
pred_svc_new=clf.predict(X_test)
print(classification_report(y_test, pred_svc_new))