import pandas as pd

import seaborn as sns 

import numpy as np 

import matplotlib.pyplot as plt

%matplotlib inline
df = pd.read_csv('../input/KNN_Project_Data')
df.head()
sns.pairplot(data=df,hue='TARGET CLASS')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
df_new = pd.DataFrame(scaled_features)

df_new.head()
from sklearn.model_selection import train_test_split

X = df_new

y = df['TARGET CLASS']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
error_rate = []

for i in range(1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))

    

    
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(X_train,y_train)

pred = knn.predict(X_test)

print(classification_report(y_test,predictions))