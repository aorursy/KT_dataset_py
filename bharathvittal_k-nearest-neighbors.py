import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('KNN_Project_Data')
df.head()
sns.pairplot(df,hue='TARGET CLASS')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = sc.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])

df_feat.head()
X = df_feat

y = df['TARGET CLASS']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
y_train
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)
np.mean(predictions!= y_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
k_value=[]

error_rate=[]





for i in range(1,40):

    k_value.append(i)

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    predictions = knn.predict(X_test)

    error_rate.append(np.mean(predictions!= y_test))
plt.figure(figsize=[10,6])

plt.plot(k_value,error_rate, color='blue',linestyle ='--',marker = 'o',markerfacecolor='red',markersize=10)

plt.xlabel('K Value')

plt.ylabel('Error rate')

plt.title('Error rate vs K value')
knn = KNeighborsClassifier(n_neighbors=28)

knn.fit(X_train,y_train)

predictions = knn.predict(X_test)

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))
knn = KNeighborsClassifier(n_neighbors=28)

knn.fit(X_train,y_train)

predictions = knn.predict(X_test)

print('With K=28' )

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))