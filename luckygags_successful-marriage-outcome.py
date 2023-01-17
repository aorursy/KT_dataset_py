import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('../input/married-at-first-sight/mafs.csv')
df.head()
df.shape
df.columns
df['Status'].hist()
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()
df['Decision'].hist()
plt.xlabel('Decision')
plt.ylabel('Count')
plt.show()
status =df.groupby('Status')

df['Age'].hist(bins = 10)
df.dtypes
df['Status'] = df['Status'].map({'Married':1,'Divorced':0})
df['Gender'] = df['Gender'].map({'M':1,'F':0})
df['Decision'] = df['Decision'].map({'Yes':1,'No':0})
df.head()
status['Decision'].plot()
corrindex = df.corr()
plt.figure(figsize=(12,12))
sns.heatmap(data = corrindex.corr(), annot=True )
df.describe(include ='all')
df.columns
df = df.drop(['Couple','Season','Location'],axis=1)
df = df.drop('Name',axis=1)
df = df.drop('Occupation',axis=1)
df.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


X = df.drop('Status', axis=1)
Y = df['Status']
X = scaler.fit_transform(X)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X,Y)
y_pred = lr.predict(X)

score = lr.score(X,Y)
score
from sklearn.metrics import confusion_matrix, classification_report
cm = classification_report(Y, y_pred)
print(cm)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.neighbors import KNeighborsRegressor

error =[]
for i in range(1,15):
   knn = KNeighborsRegressor(n_neighbors=i)
   knn.fit(x_train,y_train)
   y_pred_knn = knn.predict(x_test) 
   error.append(np.mean(y_pred_knn != y_test))

print(error)
plt.plot(range(1,15), error, color ='blue', linestyle = 'dashed')
plt.show()

  
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=25)
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
reg.score(x_train,y_train)
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import accuracy_score
classifier=ExtraTreesRegressor()

classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
classifier.score(x_train,y_train)

