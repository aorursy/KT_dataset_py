# Assignment 6
# Authors:
# Rohan Chaudhari - 01FB16ECS306
# Mehul Garg - 01FB16ECS206
# Naman Bansal - 01FB16ECS220

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import seaborn as sns
import matplotlib.pyplot as plt
#importing data
data_file = "../input/Absenteeism_at_work.csv"
df= pd.read_csv(data_file)
df.head()
df.describe()
#Preprocessing data - stage 1 (Removing outliers in label)
sns.boxplot(df['Absenteeism time in hours'])
median = np.median(df['Absenteeism time in hours'])
q75, q25 = np.percentile(df['Absenteeism time in hours'], [75 ,25])
iqr = q75 - q25
print("Lower outlier bound:",q25 - (1.5*iqr))
print("Upper outlier bound:",q75 + (1.5*iqr))
#dropping the following outliers above 17
df= df[df['Absenteeism time in hours']<=17]
df= df[df['Absenteeism time in hours']>=-7]
#Splitting data into training and testing
from sklearn.model_selection import train_test_split
y=df['Absenteeism time in hours']
X=df.drop('Absenteeism time in hours',axis=1)#Extracting only the features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
print(df.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print("Number of unique ouput classes after preprocessing:",((np.unique(y_train))))
 #Calculate the correlation of the above variables
cor = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(cor, square = True,cmap='BuGn') 
#Preprocessing - stage 2
#Dropping the following attributes due to multi-collinearity
X_train=X_train.drop('Service time',axis=1)#dropping value for training data set
X_test=X_test.drop('Service time',axis=1)#dropping value for training data set
#Preprocesssing - stage 3 (Normalizing features)
from sklearn import preprocessing
X_scaled_train = preprocessing.scale(X_train)
X_scaled_test = preprocessing.scale(X_test)
X_scaled_train.shape
#Model 1 : KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report

#We kept ID attribute as we observed that ID was repeated and had a pattern with labels
knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(X_scaled_train, y_train)
y_pred = knn.predict(X_scaled_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("------------------------\n")
print(classification_report(y_test, y_pred))

#Model 2 - SVM (SV classifier)
from sklearn import metrics, svm
from sklearn.svm import SVC

svm=svm.SVC()
svm.fit(X_scaled_train, y_train)
y_pred = svm.predict(X_scaled_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("------------------------\n")
print(classification_report(y_test, y_pred))
#Model 3 - Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier 

dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_scaled_train, y_train) 
y_pred = dtree_model.predict(X_scaled_test) 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("------------------------\n")
print(classification_report(y_test, y_pred))
#Model 4 - Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB().fit(X_scaled_train, y_train) 
y_pred = gnb.predict(X_scaled_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("------------------------\n")
print(classification_report(y_test, y_pred))
#Model 5 - Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier().fit(X_scaled_train, y_train)
y_pred = rf.predict(X_scaled_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("------------------------\n")
print(classification_report(y_test, y_pred))
#Model 6 - Multi Layer Perceptron
from sklearn.neural_network import MLPClassifier

mlp=MLPClassifier(max_iter=4000, alpha=0.1).fit(X_scaled_train,y_train)
y_pred = mlp.predict(X_scaled_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("------------------------\n")
print(classification_report(y_test, y_pred))
##Conclusion

#Model 1 - SVM SVC - 52.85% Accuracy
# Justification - The kernel used for SVC is Radial Basis Function(RBF).
#One property of the RBF kernel is that it is infinitely smooth
#They are relatively easy to calibrate, as opposed to other kernels.
#It has localized and finite response along the entire x-axis.

#Model 2 - MLP - 50% Accuracy 
#Maximum Convergence occured at epoch value of 4000.
#Alpha-value of 0.1 provides an optimum accuracy as we found by trial-and-error.


#SVM model is better than MLP due to the following observations
#1.Higher Overall Accuracy
#2. Higher Overall Average Precison over majority of output classes
#3. Higher Overall F1-Score over majority of Output classes.
#Also,
#4.Takes lesser computation power(time).
#5.Simpler and better for a multi-class classification problem

