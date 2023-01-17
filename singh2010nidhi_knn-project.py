import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv("../input/knn-data1/KNN_Project_Data")
df.head()
sns.pairplot(df,hue="TARGET CLASS",palette="coolwarm")
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features=scaler.transform(df.drop('TARGET CLASS',axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])

df_feat.head()
from sklearn.model_selection import train_test_split
X = df_feat

y = df['TARGET CLASS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
error_rate = []

for i in range(1,60):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))

plt.plot(range(1,60),error_rate,color='blue',linestyle="--",marker="o", markerfacecolor='red',markersize=10)

plt.title("Error Rate Vs K")

plt.xlabel("K")

plt.ylabel("Error Rate")
knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(X_train, y_train)

pred = knn.predict(X_test)

print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))