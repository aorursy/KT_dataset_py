import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
df = pd.read_csv("../input/Classified Data",index_col=0)
df.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
scaled_features
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])

df_feat.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],

                                                    test_size=0.30)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
pred
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
error_rate = []



# Will take some time

for i in range(1,40):

    

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
error_rate
plt.figure(figsize=(10,6))

plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',

         markerfacecolor='red', markersize=10)

plt.title('Error Rate vs. K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
# FIRST A QUICK COMPARISON TO OUR ORIGINAL K=1

knn = KNeighborsClassifier(n_neighbors=1)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=1')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
# NOW WITH K=23

knn = KNeighborsClassifier(n_neighbors=23)



knn.fit(X_train,y_train)

pred = knn.predict(X_test)



print('WITH K=23')

print('\n')

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))