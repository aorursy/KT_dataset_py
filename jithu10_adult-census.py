import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


#Module for resampling
from sklearn.utils import resample

#for model creation
from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#for decision tree classifcation
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


#for randomforest classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#for knn classification
from sklearn.neighbors import KNeighborsClassifier

#for svm classification
from sklearn.svm import SVC
from sklearn import svm

#for mlp classification
from sklearn.neural_network import MLPClassifier


data=pd.read_csv('../input/adult-census-income/adult.csv')
data.head(10)
data.info()
#Removal of missing data specified by question mark(?)
#New dataframe df
df = data[(data != '?').all(axis=1)]
df.head()
df.tail()
# Binary encoding of the target variable
df['income'] = df['income'].apply(lambda inc: 0 if inc == "<=50K" else 1) 
#One-hot encoding of the categorical columns
#converting categorical data to binary
df = pd.get_dummies(df,columns=['workclass','sex', 'marital.status',
                                    'race','relationship','occupation'],
               prefix=['workclass', 'is', 'is', 'race_is', 'relation', 'is'], drop_first=True)
df.head()
print(df['income'].value_counts())
#Balancing data
# Separate majority and minority classes
df_majority = df[df.income==0]
df_minority = df[df.income==1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=22654,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.income.value_counts()
df_upsampled.info()
plt.figure(figsize=(20,12))
sns.heatmap(df_upsampled.corr())
#Splitting dataset into training and testing class
array = df_upsampled.values
X = array[:,0:8 and 9:44]
Y = array[:,8]
Y=Y.astype('int')

X_train,X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=0.20, random_state=1)
# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy",max_depth=3)

clf = clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_validation)

cm = confusion_matrix(Y_pred, Y_validation)

print("Accuracy of Decision tree classification:",metrics.accuracy_score(Y_validation, Y_pred))
print(cm)
#decision tree
plt.figure(figsize=(10,10))
tree.plot_tree(clf)
#random forest classification
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf = clf.fit(X_train,Y_train)

Y_pred = clf.predict(X_validation)
cm = confusion_matrix(Y_pred, Y_validation)

print("Accuracy random froest classification:",metrics.accuracy_score(Y_validation, Y_pred))
print(cm)
#knn clasification
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train,Y_train)

Y_pred = knn.predict(X_validation)
cm = confusion_matrix(Y_pred, Y_validation)

print("Accuracy of knn classification:",metrics.accuracy_score(Y_validation, Y_pred))
print(cm)
#support vector machine Classification
clf = svm.SVC(kernel='linear') 

clf.fit(X_train, Y_train)


Y_pred = clf.predict(X_validation)
cm = confusion_matrix(Y_pred, Y_validation)


print("Accuracy of SVM Classifier : ",metrics.accuracy_score(Y_validation, Y_pred))
print(cm)


#MultilayerPercetron classification

clf = MLPClassifier(hidden_layer_sizes=(3,3), max_iter=3000,activation = 'relu',solver='adam',random_state=1)
clf=clf.fit(X_train, Y_train)

cm = confusion_matrix(Y_pred, Y_validation)
Y_pred = clf.predict(X_validation)

print("Accuracy of MLPClassifier : ",metrics.accuracy_score(Y_validation, Y_pred))
print(cm)
