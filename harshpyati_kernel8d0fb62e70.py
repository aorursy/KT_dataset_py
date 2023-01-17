import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
df = pd.read_csv('../input/voice.csv')
df.info()
df.head(5)
df.isna().values.any()
df.min()
df.max()
df.head(10)
df.tail(10)
label = [1 if each == "male" else 0 for each in df.label]
label = pd.Series(np.array(label))
label = pd.DataFrame(label,columns=['gender'])
df.drop('label',axis = 1,inplace = True)
df = pd.concat([df,label],axis = 1)
df = df.sample(frac = 1).reset_index(drop = True) ## drop = True, prevents df from creating another column with the old index
df.tail(10)
df.columns
df['gender'].values
X = df.drop('gender',axis = 1)
y = df['gender']
X = (X - X.min())/(X.max() - X.min())
X.min()
X.max()
X.head(10)
from sklearn.model_selection import train_test_split
X.shape
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1)
X_train.shape
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
pred = lr.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))
from sklearn.neighbors import KNeighborsClassifier
error_rate = []

for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors = i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_i = np.mean(pred_i!=y_test)
    error_rate.append(error_i)
plt.figure(figsize=(10,6))
plt.plot(range(1,10),error_rate,color='green', linestyle='dashed', marker='o',
         markerfacecolor='yellow', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors = 8)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(classification_report(y_test,pred))
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train,y_train)
pred = svc.predict(X_test)
print(classification_report(y_test,pred))
from sklearn.tree import DecisionTreeClassifier
dec = DecisionTreeClassifier()
dec.fit(X_train,y_train)
pred = dec.predict(X_test)
print(classification_report(y_test,pred))
from sklearn.ensemble import RandomForestClassifier
error_rate = []

for i in range(1,11):
    rfc = RandomForestClassifier(n_estimators = i)
    rfc.fit(X_train,y_train)
    pred_i = rfc.predict(X_test)
    error_rate.append(np.mean(y_test!=pred_i))
plt.figure(figsize=(10,6))
plt.plot(range(1,11),error_rate,color='pink', linestyle='dashed', marker='o',
         markerfacecolor='black', markersize=10)
plt.title('Error Rate vs. No.of Estimators')
plt.xlabel('No.of Estimators')
plt.ylabel('Error Rate')
rfc = RandomForestClassifier(n_estimators = 5)
rfc.fit(X_train,y_train)
pred = rfc.predict(X_test)
print(classification_report(y_test,pred))
import xgboost as xgb
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train,y_train)
pred = xgb_model.predict(X_test)
print(classification_report(y_test,pred))
nEstimators = [10,20,30,40,50,60,70,80,90,100,110,120]
error_rate = []
for i in nEstimators:
    xgb_model = xgb.XGBClassifier(n_estimators=i)
    xgb_model.fit(X_train,y_train)
    pred = xgb_model.predict(X_test)
    error_rate.append(np.mean(y_test!=pred))
plt.figure(figsize=(10,6))
plt.plot(range(1,13),error_rate,color='black', linestyle='dashed', marker='o',
         markerfacecolor='orange', markersize=10)
plt.title('Error Rate vs. No.of Estimators')
plt.xlabel('No.of Estimators')
plt.ylabel('Error Rate')
xgb_model = xgb.XGBClassifier(n_estimators = 120)
xgb_model.fit(X_train,y_train)
pred = xgb_model.predict(X_test)
print(classification_report(y_test,pred))
