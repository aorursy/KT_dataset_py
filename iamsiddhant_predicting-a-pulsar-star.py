import pandas as pd
data=pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')
data.head()
data.columns
data.isnull().sum()
data.info()
data.describe()
data['target_class'].unique()
import seaborn as sns

import matplotlib.pyplot as plt
sns.countplot(data['target_class'])
data[data['target_class']==1].count()
data[data['target_class']==0].count()
data.head()
correlation=data.corr()

fig = plt.figure(figsize=(12, 10))



sns.heatmap(correlation, annot=True, center=1)
y=data['target_class']
X=data.drop(['target_class'],axis=1)
X.head()
X.describe()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
model.fit(X_train,y_train)
model.score(X_test,y_test)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train, y_train)

pred_rfc = rfc.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, pred_rfc))
rfc.score(X_test,y_test)
from sklearn.svm import SVC
clf = SVC() 
clf.fit(X_train, y_train) 
pred_svc=clf.predict(X_test)
print(classification_report(y_test, pred_svc))
clf.score(X_test,y_test)
print('Accuracy from Support Vector Classification is '+str(clf.score(X_test,y_test)*100)+"%")

print('Accuracy from Random forest classifier is '+str(rfc.score(X_test,y_test)*100)+'%')

print('Accuracy from Logistic regression is '+str(model.score(X_test,y_test)*100)+'%')