import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

%matplotlib inline
df = pd.read_csv("../input/Classified Data.csv",index_col = 0)
df.head()
#Target class and random letters....annoymous data.
#Use features to predict target class
#Scale matters a lot since trying to predict closest to....
#Standardize all to be on same scale so easier to figure out
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
#Fit to data...All feature columns fit data to. Use for transformation
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))
#Transform performs by centering and scaling. 
scaled_features

#Array of values - Scaled of the dataframe
#Scaled_Features passing the data above into this.
df.columns[:-1]

#Everything but last one
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()
#Ready for KNN
from sklearn.model_selection import train_test_split
X = df_feat

y= df['TARGET CLASS']

#FOR X could use df_feat or the scaled_features array
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.neighbors import KNeighborsClassifier
#Will they be in target class or not? Start with K=1 and use elbow method
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
#Passing in test data
pred = knn.predict(X_test)
pred
#Predictions of what class they belong to based on features
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))
#Use elbow method to choose correct k value. 92% accuracy approx
error_rate = []
# Empty list gonna try different tests to see lowest error rate
#EVery possible k value 1-40 ... could use less
for i in range (1,40):

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    error_rate.append(np.mean(pred_i != y_test))
#Error rate...append mean of pred_i..average error rate. not equl to actual
#test values
error_rate
plt.plot(range(1,40),error_rate,color='blue',linestyle='dashed',marker='o', markerfacecolor='red',markersize=10)

plt.title('Error Rate vs K Value')

plt.xlabel('K')

plt.ylabel('Error Rate')
#Higher error rate with higher k. Min around 20s...K 35 or so might be better
#Choosing around 17 because eventhough higher going lower the bouncing 
#aroud isn't great in the 30s...any of these tho have low error rates
#Comparing number of numbers 17 to 1 above to see accuracy

knn = KNeighborsClassifier(n_neighbors = 17)

knn.fit(X_train,y_train)

pred= knn.predict(X_test)

print(confusion_matrix(y_test,pred))

print('\n')

print(classification_report(y_test,pred))
#Performance approx 95% with the elbox method. Had to spend more time to
#Find optimal but got a couple more percent better...