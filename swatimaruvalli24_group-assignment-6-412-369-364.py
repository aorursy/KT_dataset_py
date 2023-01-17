#ASSIGNMENT NO. 6
#SWATI.MARUVALLI 01FB16ECS412
#SHREYA ESWARAIAH 01FB16ECS369
#SIRISHA R RAO 01FB16ESC364

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
import seaborn as sns
import matplotlib.pyplot as plt

data_file = "../input/Absenteeism_at_work.csv"
df= pd.read_csv(data_file)
df.head()
df.describe()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
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
sns.heatmap(cor, square = True,cmap='viridis')
#Splitting data into training and testing
from sklearn.model_selection import train_test_split
y=df['Absenteeism time in hours']
X=df.drop('Absenteeism time in hours',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
#scaling the data
from sklearn import preprocessing
X_scaled_train = preprocessing.scale(X_train)
X_scaled_test = preprocessing.scale(X_test)
X_scaled_train.shape

#KNN
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
#SVM (SV classifier)
from sklearn import metrics, svm
from sklearn.svm import SVC

svm=svm.SVC()
svm.fit(X_scaled_train, y_train)
y_pred = svm.predict(X_scaled_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("------------------------\n")
print(classification_report(y_test, y_pred))
#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier 

dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_scaled_train, y_train) 
y_pred = dtree_model.predict(X_scaled_test) 
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("------------------------\n")
print(classification_report(y_test, y_pred))
#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB().fit(X_scaled_train, y_train) 
y_pred = gnb.predict(X_scaled_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("------------------------\n")
print(classification_report(y_test, y_pred))

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier().fit(X_scaled_train, y_train)
y_pred = rf.predict(X_scaled_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("------------------------\n")
print(classification_report(y_test, y_pred))
# Multi Layer Perceptron
from sklearn.neural_network import MLPClassifier

mlp=MLPClassifier(max_iter=4000, alpha=0.1).fit(X_scaled_train,y_train)
y_pred = mlp.predict(X_scaled_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("------------------------\n")
print(classification_report(y_test, y_pred))

#SVM SVC - 52.85% Accuracy
# The kernel used for SVC is Radial Basis Function(RBF).
#One property of the RBF kernel is that it is infinitely smooth
#They are relatively easy to calibrate, as opposed to other kernels.
#It has localized and finite response along the entire x-axis.

#MLP - 50% Accuracy 
#Maximum Convergence occured at epoch value of 4000.
#Alpha-value of 0.1 provides an optimum accuracy as we found by trial-and-error.


#SVM model is better than MLP due to the following observations
#1.Higher Overall Accuracy
#2. Higher Overall Average Precison over majority of output classes
#3. Higher Overall F1-Score over majority of Output classes.
#Also,
#4.Takes lesser computation power(time).
#5.Simpler and better for a multi-class classification problem
#The accuracy for Decision tree classifier is the highest amongst all other models.