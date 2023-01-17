import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
dataframe = pd.read_csv("../input/knn-data1/KNN_Project_Data")
dataframe.head()
sns.pairplot(dataframe, hue="TARGET CLASS")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(dataframe.drop("TARGET CLASS", axis=1))
scaled_feat = scaler.transform(dataframe.drop("TARGET CLASS", axis=1))
scaled_feat
df_feat = pd.DataFrame(scaled_feat, columns=dataframe.columns[:-1])
df_feat.head()
from sklearn.model_selection import train_test_split
X = df_feat
y = dataframe["TARGET CLASS"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_new = knn.predict(X_test)
    error_rate.append(np.mean(pred_new != y_test))
plt.figure(figsize=(15,4))
plt.plot(range(1,40), error_rate)
knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
pred_new = knn.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))