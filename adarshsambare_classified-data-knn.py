# Importing Required Libraries
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
print(os.listdir("../input"))

# Reading the data
df = pd.read_csv("../input/classified-data/Classified Data")
# Checking the data
df.head()
# Data info
df.info()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Scaling the whole data except Y
scaler.fit(df.drop('TARGET CLASS', axis = 1))
# Saving the transformed data
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis = 1))
# Saving the scaled feature in new dataframe
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
from sklearn.model_selection import train_test_split
x = df_feat
y = df['TARGET CLASS']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 101)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
# Fitting the model on training data
knn.fit(x_train,y_train)
# Predicting on test data
pred = knn.predict(x_test)
# For accuracy 
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))
# 91% 
# we have done a good job
error_rate = []
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train )
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(12,12))
plt.plot(range(1,40), error_rate, color = 'blue', 
linestyle='dashed', marker = "o",
markerfacecolor = 'red',markersize = 10)
plt.title('Error Rate Vs Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
# Good accuracy for KNN = 11
# then directly to 30
knn1 = KNeighborsClassifier(n_neighbors=11)
#knn2 = KNeighborsClassifier(n_neighbors=18)

knn1.fit(x_train,y_train)
#knn2.fit(x_train,y_train)

pred1 = knn1.predict(x_test)
#pred2 = knn2.predict(x_test)

print(confusion_matrix(pred1,y_test))
print(classification_report(pred1,y_test))

#print(confusion_matrix(pred2,y_test))
#print(classification_report(pred2,y_test))
# 94 % Accuracy