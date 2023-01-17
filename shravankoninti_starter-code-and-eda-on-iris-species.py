# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt                                      # to plot graph
import seaborn as sns                                                # for intractve graphs
%matplotlib inline
from sklearn.linear_model import LogisticRegression                  # for Logistic regression
from sklearn.cross_validation import train_test_split                # to split the data
from sklearn.metrics import accuracy_score                           #Accuracy score calculation
from sklearn.metrics import classification_report, confusion_matrix  # Classification report and confusion matrix
from sklearn.neighbors import KNeighborsClassifier                   # for K nearest neighbours
from sklearn import svm                                              #for Support Vector Machine (SVM) Algorithm
from sklearn.tree import DecisionTreeClassifier                      #for using Decision Tree Algoithm



# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# df = pd.read_csv('../input/train.csv')
df = pd.read_csv('../input/Iris.csv')
df.head()
print("the total nuumber of rows {} and columns {} are present in this dataset :".format(df.shape[0], df.shape[1]))
# First we will identify the predictor and target variable
df.columns
#Check the data types of each column
df.dtypes
num_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
cat_cols = ['Species']
# Check the levels in Speciies
df['Species'].value_counts()
#Histogram plots for knowing the data distribution
plt.figure(1)
plt.figure(figsize = (15,10))
num_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for col in num_cols:
    plt.subplot(int(str(22)+str((num_cols.index(col)+1))))
    sns.distplot(df[col])      
   
#Box plots - shows visual outliers in the data
plt.figure(1)
plt.figure(figsize = (15,10))
num_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for col in num_cols:
    plt.subplot(int(str(22)+str((num_cols.index(col)+1))))  
    sns.boxplot(x=col,  data=df)

   
# Check for any class Imbalance
df['Species'].value_counts(normalize = True)
df['Species'].value_counts().plot.bar(figsize=(10,4),title='Species - Split for Dataset')
plt.xlabel('Species')
plt.ylabel('Count')
plt.figure(1)
plt.figure(figsize = (12,8))
plt.scatter(df['SepalLengthCm'], df['SepalWidthCm'], s=np.array(df.Species == 'Iris-setosa'), marker='^', c='green', linewidths=5)
plt.scatter(df['SepalLengthCm'], df['SepalWidthCm'], s=np.array(df.Species == 'Iris-versicolor'), marker='^', c='orange', linewidths=5)
plt.scatter(df['SepalLengthCm'], df['SepalWidthCm'], s=np.array(df.Species == 'Iris-virginica'), marker='o', c='blue', linewidths=5)
plt.xlabel('SepalLengthCm')
plt.ylabel('SepalWidthCm')
plt.legend(loc = 'upper left', labels =['Setosa', 'versicolor', 'virginica'])
plt.show()
plt.figure(1)
plt.figure(figsize = (12,8))
plt.scatter(df['PetalLengthCm'], df['PetalWidthCm'], s=np.array(df.Species == 'Iris-setosa'), marker='^', c='green', linewidths=5)
plt.scatter(df['PetalLengthCm'], df['PetalWidthCm'], s=np.array(df.Species == 'Iris-versicolor'), marker='^', c='orange', linewidths=5)
plt.scatter(df['PetalLengthCm'], df['PetalWidthCm'], s=np.array(df.Species == 'Iris-virginica'), marker='o', c='blue', linewidths=5)
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.legend(loc = 'upper left', labels = ['Setosa', 'versicolor', 'virginica'])
plt.show()
# Correlation between numerical variables
num_cols_data = (df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
matrix = num_cols_data.corr()
f, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(matrix, vmax=.8, square=True, cmap="YlGnBu", annot = True);
#Pair Plot
sns.set()
columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
sns.pairplot(df[columns],size = 3 ,kind ='scatter',diag_kind='kde')
plt.show()
plt.figure(1)
plt.figure(figsize = (15,10))
num_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
for col in num_cols:
    plt.subplot(int(str(22)+str((num_cols.index(col)+1))))
    sns.violinplot(x='Species', y = col, data = df)   
#Check for missing values 
df.isnull().sum()
X = df.drop(['Id','Species'],1)
y = df.Species
# TODO: Shuffle and split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Success
print ("Training and testing split was successful.")

model_log = LogisticRegression()
model_log.fit(X_train, y_train)
pred_cv_train = model_log.predict(X_train)
pred_cv_test = model_log.predict(X_test)
print("the accuracy for train data is {}".format(accuracy_score(y_train,pred_cv_train)))
print("the accuracy for test data is {}".format(accuracy_score(y_test,pred_cv_test)))
confusion_matrix = confusion_matrix( y_test,pred_cv_test)
print("the recall for this model is :",confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[1,0]))

fig= plt.figure(figsize=(6,3))# to plot the graph
print("TP",confusion_matrix[1,1,]) 
print("TN",confusion_matrix[0,0]) 
print("FP",confusion_matrix[0,1]) 
print("FN",confusion_matrix[1,0]) 
sns.heatmap(confusion_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)
plt.title("Confusion_matrix")
plt.xlabel("Predicted_class")
plt.ylabel("Real class")
plt.show()
print(confusion_matrix)
print("\n--------------------Classification Report------------------------------------")
print(classification_report(y_test, pred_cv_test)) 
model_svm = svm.SVC() 
model_svm.fit(X_train,y_train) 
pred_cv_train = model_svm.predict(X_train)
pred_cv_test = model_svm.predict(X_test)
print("the accuracy for train data is {}".format(accuracy_score(y_train,pred_cv_train)))
print("the accuracy for test data is {}".format(accuracy_score(y_test,pred_cv_test)))
model_dt=DecisionTreeClassifier()
model_dt.fit(X_train,y_train)
pred_cv_train = model_dt.predict(X_train)
pred_cv_test = model_dt.predict(X_test)
print("the accuracy for train data is {}".format(accuracy_score(y_train,pred_cv_train)))
print("the accuracy for test data is {}".format(accuracy_score(y_test,pred_cv_test)))
model_knn=KNeighborsClassifier(n_neighbors=3) #this examines 3 neighbours as one pool, for putting the new data into a class
model_knn.fit(X_train,y_train)
pred_cv_train = model_knn.predict(X_train)
pred_cv_test = model_knn.predict(X_test)
print("the accuracy for train data is {}".format(accuracy_score(y_train,pred_cv_train)))
print("the accuracy for test data is {}".format(accuracy_score(y_test,pred_cv_test)))
# We can identify the accuracy for which K-value this model gives best accuracy
index=list(range(1,15))
accur=pd.Series()
x=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
for i in list(range(1,15)):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(X_train,y_train)
    pred_test=model.predict(X_test)
    accur=accur.append(pd.Series(accuracy_score(pred_test,y_test)))
plt.plot(index, accur)
plt.xticks(x);