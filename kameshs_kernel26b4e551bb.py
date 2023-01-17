import numpy as np

import pandas as pd

import sklearn

import scipy

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
#Load data set

dataFrame = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
#number of rows and columns

dataFrame.shape
dataFrame.head(5)
dataFrame.isnull().values.any()
dataFrame.describe()
count_by_class = pd.value_counts(dataFrame['Class'], sort = True)



## % of Fraud observations

print("")

print("% of Fraud (outlier) cases in dataset =",(count_by_class[1]/(count_by_class[1]+count_by_class[0])) * 100)

print("")



##Graphical view

count_by_class.plot(kind='bar', color="g")

plt.title("Distribution of Normal Vs Fraud observations in dataset.")

plt.xlabel("Classes")

plt.xticks(range(2), ["Normal", "Fraud"])

plt.ylabel("Number of observations");
f, (ax1, ax2) = plt.subplots(2, sharex=True, sharey=True)

plt.xlabel('Time of Transaction')

plt.ylabel('Amount in Transaction')



fraud_observations = dataFrame[dataFrame['Class']==1]

ax1.scatter(fraud_observations.Time, fraud_observations.Amount, color='r')

ax1.set_title('Fraud')



normal_observations = dataFrame[dataFrame['Class']==0]

ax2.scatter(normal_observations.Time, normal_observations.Amount, color='g')

ax2.set_title('Normal')





plt.show()
correlation_matrix = dataFrame.corr()

print(correlation_matrix)
fig = plt.figure(figsize=(31,31))

sns.heatmap(correlation_matrix)

plt.show()
from sklearn.model_selection import train_test_split

y = dataFrame['Class']

X = dataFrame.drop('Class', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print (" Training data set : ",X_train.shape)

print (" Test data set : ",X_test.shape)

print (" Number of Fraud cases in training set : ",len(y_train[y_train==1]))

print (" Number of Fraud cases in test set : ",len(y_test[y_test==1]))
from sklearn.metrics import classification_report,accuracy_score

from sklearn.ensemble import IsolationForest



fraud_fraction = len(y_train[y_train==1])/float(len(y_train[y_train==0]))

number_of_train_samples=len(X_train)



# fit the model

clf=IsolationForest(n_estimators=100, max_samples=number_of_train_samples,contamination=fraud_fraction,random_state=None, verbose=0)

clf.fit(X_train)



#predict on training set

y_pred_train = clf.predict(X_train)



#predict on test set

y_pred_test = clf.predict(X_test)



#Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions

y_pred_train[y_pred_train == 1] = 0

y_pred_train[y_pred_train == -1] = 1



y_pred_test[y_pred_test == 1] = 0

y_pred_test[y_pred_test == -1] = 1





print("-----------------------------------------------------------")

print(" Score on Test set")

print(" Error count : ",(y_test!=y_pred_test).sum())

print(" Accuracy Score:")

print(accuracy_score(y_test,y_pred_test))

print(" Classification Report:")

print(classification_report(y_test,y_pred_test))



print("-----------------------------------------------------------")

print(" Score on training set")

print(" Error count : ",(y_train != y_pred_train).sum())

print(" Accuracy Score:")

print(accuracy_score(y_train,y_pred_train))

print(" Classification Report:")

print(classification_report(y_train,y_pred_train))
from sklearn.metrics import classification_report,accuracy_score

from sklearn.neighbors import LocalOutlierFactor



fraud_fraction = len(y_train[y_train==1])/float(len(y_train[y_train==0]))

number_of_train_samples=len(X_train)



# fit the model

clf=LocalOutlierFactor(n_neighbors=20, algorithm='auto',leaf_size=30, metric='minkowski',p=2, metric_params=None, contamination=fraud_fraction)

clf.fit(X_train)





#predict on training set

y_pred_train = clf.fit_predict(X_train)



#predict on test set

y_pred_test = clf.fit_predict(X_test)





#Reshape the prediction values to 0 for Valid transactions , 1 for Fraud transactions

y_pred_train[y_pred_train == 1] = 0

y_pred_train[y_pred_train == -1] = 1



y_pred_test[y_pred_test == 1] = 0

y_pred_test[y_pred_test == -1] = 1





print("-----------------------------------------------------------")

print(" Score on Test set")

print(" Error count : ",(y_test!=y_pred_test).sum())

print(" Accuracy Score:")

print(accuracy_score(y_test,y_pred_test))

print(" Classification Report:")

print(classification_report(y_test,y_pred_test))



print("-----------------------------------------------------------")

print(" Score on training set")

print(" Error count : ",(y_train != y_pred_train).sum())

print(" Accuracy Score:")

print(accuracy_score(y_train,y_pred_train))

print(" Classification Report:")

print(classification_report(y_train,y_pred_train))