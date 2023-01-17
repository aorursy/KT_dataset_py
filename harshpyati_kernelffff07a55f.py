import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/heart.csv')
df.shape
df.columns
import seaborn as sns
sns.countplot(x = 'sex',data = df)  # 1-Male 0-Female
sns.countplot(x = 'sex',data = df,hue='target') # 1-Male, 0-Female
sns.heatmap(df.corr(),cmap = 'Blues')
plt.title('Correlation B/w Various Features')
df['cp'].unique()
sns.countplot(x = 'cp',data = df)
sns.countplot(x = 'cp',data = df,hue = 'target')
plt.title('Chest Pain Type vs Target')
df['thalach'].max()
df['thalach'].min()
df['thalach'].mean()
df[df['thalach'] == df['thalach'].max()]
df[df['thalach'] == df['thalach'].min()]
sns.swarmplot(y = 'thalach',x = 'target',data=df)
sns.boxplot(x = 'target',y = 'age',data = df)
X = df.drop('target',axis = 1)
y = df['target']
X.shape
y.shape
X.columns
X.sex.unique()
sex = pd.get_dummies(X['sex'],prefix = 'Gender')
sex = sex.rename(columns = {
    "Gender_0": "Female",
    "Gender_1": "Male"
})
X.drop('sex',inplace=True,axis = 1)
X.sex = sex
X = pd.concat([sex,X],axis = 1)
X.columns
chestPainType = pd.get_dummies(X.cp,prefix = 'Type')
X.drop('cp',axis = 1,inplace=True)
X = pd.concat([X,chestPainType],axis = 1)
X.columns
exang = pd.get_dummies(X.exang,prefix = 'Exang')
X.drop('exang',axis = 1,inplace=True)
X = pd.concat([X,exang],axis = 1)
X.columns
X.ca.unique()
X.head(5)
from sklearn.preprocessing import StandardScaler
sca = StandardScaler()
sca.fit(X)
X = sca.transform(X)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.131)
X_train.shape
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train,y_train)
pred = log.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(classification_report(y_test,pred))
### Choosing the appropriate no.of neighbors
error_rate = []

for i in range(1,10):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,10),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
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
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train,y_train)
pred = rfc.predict(X_test)
print(classification_report(y_test,pred))
error_rate = []
for i in range(1,10):
    rfc = RandomForestClassifier(n_estimators=i)
    rfc.fit(X_train,y_train)
    pred_i = rfc.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,10),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. No.of Estimators')
plt.xlabel('No.of Estimators')
plt.ylabel('Error Rate')
rfc = RandomForestClassifier(n_estimators = 7)
rfc.fit(X_train,y_train)
pred = rfc.predict(X_test)
print(classification_report(y_test,pred))
