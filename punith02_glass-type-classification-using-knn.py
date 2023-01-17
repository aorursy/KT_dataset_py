import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df=pd.read_csv('../input/glass/glass.csv')
df.head()
df.info()
df.describe()
plt.figure(figsize=(10,8))

sns.heatmap(df.corr(),annot=True,cmap='coolwarm')
sns.stripplot(x='Type',y='RI',data=df)
sns.stripplot(x='Type',y='Na',data=df)
sns.stripplot(x='Type',y='Mg',data=df)
sns.stripplot(x='Type',y='Al',data=df)
sns.stripplot(x='Type',y='Si',data=df)
sns.stripplot(x='Type',y='K',data=df)
sns.stripplot(x='Type',y='Ca',data=df)
sns.stripplot(x='Type',y='Ba',data=df)
sns.stripplot(x='Type',y='Fe',data=df)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

scaler.fit(df.drop('Type',axis=1))
scaled_features=scaler.transform(df.drop('Type',axis=1))

df_head=pd.DataFrame(scaled_features,columns=df.columns[:-1])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df_head,df['Type'], test_size=0.3, random_state=40)
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)

knn.fit(x_train,y_train)
pred=knn.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print('Classification Report ',classification_report(y_test,pred))
print('Confusion Matrix',confusion_matrix(y_test,pred))
error_rate=[]

for i in range(1,40):

    knn=KNeighborsClassifier(n_neighbors=i)

    knn.fit(x_train,y_train)

    pred_i=knn.predict(x_test)

    error_rate.append(np.mean(pred_i!=y_test))

    
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('error_rate vs. K Value')

plt.xlabel('K')

plt.ylabel('error_rate')
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
print('Classification Report ',classification_report(y_test,pred))
print('Confusion Matrix',confusion_matrix(y_test,pred))