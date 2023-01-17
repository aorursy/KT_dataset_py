#Data setup

import pandas as pd



df = pd.read_csv('iris.csv')

df.head()
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values='NaN', strategy='median', axis=0)

imputer = imputer.fit(df.iloc[:,:-1])

imputed_data = imputer.transform(df.iloc[:,:-1].values)

df.iloc[:,:-1] = imputed_data



iris = df
iris.iloc[:,5].unique()
iris.head()
from sklearn.preprocessing import LabelEncoder

class_label_encoder = LabelEncoder()



iris.iloc[:,-1] = class_label_encoder.fit_transform(iris.iloc[:,-1])
iris.head()
iris.corr()
iris.var()
splt = pd.plotting.scatter_matrix(iris, c=iris.iloc[:,-1], figsize=(20, 20), marker='o')
import numpy as np

from sklearn.model_selection import train_test_split



# Transform data into features and target

X = np.array(iris.ix[:, 1:5]) 

y = np.array(iris['Species'])



# split into train and test

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)



print(X_train.shape)

print(y_train.shape)
print(X_test.shape)

print(y_test.shape)
# loading library

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



# instantiate learning model (k = 3)

knn = KNeighborsClassifier(n_neighbors = 3)



# fitting the model

knn.fit(X_train, y_train)



# predict the response

y_pred = knn.predict(X_test)



# evaluate accuracy

print(accuracy_score(y_test, y_pred))



# instantiate learning model (k = 5)

knn = KNeighborsClassifier(n_neighbors=5)



# fitting the model

knn.fit(X_train, y_train)



# predict the response

y_pred = knn.predict(X_test)



# evaluate accuracy

print(accuracy_score(y_test, y_pred))



# instantiate learning model (k = 9)

knn = KNeighborsClassifier(n_neighbors=9)



# fitting the model

knn.fit(X_train, y_train)



# predict the response

y_pred = knn.predict(X_test)



# evaluate accuracy

print(accuracy_score(y_test, y_pred))
# creating odd list of K for KNN

myList = list(range(1,25))



# subsetting just the odd ones

neighbors = list(filter(lambda x: x % 2 != 0, myList))



# empty list that will hold accuracy scores

ac_scores = []



# perform accuracy metrics for values from 1,3,5....19

for i,k in enumerate(neighbors):

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    # predict the response

    y_pred = knn.predict(X_test)

    # evaluate accuracy

    scores = accuracy_score(y_test, y_pred)

    ac_scores.append(scores)



# changing to misclassification error

MSE = [1 - x for x in ac_scores]



# determining best k

optimal_k = neighbors[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d" % optimal_k)
import matplotlib.pyplot as plt

# plot misclassification error vs k

plt.plot(neighbors, MSE)

plt.xlabel('Number of Neighbors K')

plt.ylabel('Misclassification Error')

plt.show()
#Load all required library

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

from sklearn import datasets

from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
# Load using input file

iris=pd.read_csv("iris.csv")

iris.head(5)
# Check dimension of data

iris.shape
#Check shape of data

iris.info()
# check for missing values
iris.isna().sum()
iris = iris.dropna()
iris.isna().sum()
X=iris.iloc[:,:4].values

y=iris['Species'].values
#Check the dataset

print(y)

print(X)
iris["Species"].value_counts()
pd.value_counts(iris["Species"]).plot(kind="bar")
spd = pd.plotting.scatter_matrix(iris, figsize=(20,20), diagonal="kde")
corr = iris.corr()

corr

#Please note, it's Require to remove correlated features because they are voted twice in the model and it can lead to over inflating importance.We will ignore it here
iris
from sklearn.datasets import load_iris

iris = load_iris()



from matplotlib import pyplot as plt



# The indices of the features that we are plotting

x_index = 0

y_index = 1



# this formatter will label the colorbar with the correct target names

formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])



plt.figure(figsize=(8, 8))

plt.scatter(iris.data[:, x_index], iris.data[:, y_index], c=iris.target)

plt.colorbar(ticks=[0, 1, 2], format=formatter)

plt.xlabel(iris.feature_names[x_index])

plt.ylabel(iris.feature_names[y_index])



plt.tight_layout()

plt.show()
### SPLITTING INTO TRAINING AND TEST SETS

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20,random_state=22)
### NORMALIZTION / FEATURE SCALING

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
### WE WILL FIT THE THE CLASSIFIER TO THE TRAINING SET

naiveClassifier=GaussianNB()

naiveClassifier.fit(X_train,y_train)
y_pred = naiveClassifier.predict(X_test)
#Keeping the actual and predicted value side by side

y_compare = np.vstack((y_test,y_pred)).T

#Actual->LEFT

#predicted->RIGHT

#Number of values to be print

y_compare[:20,:]
# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
#finding accuracy from the confusion matrix.

a = cm.shape

correctPrediction = 0

falsePrediction = 0



for row in range(a[0]):

    for c in range(a[1]):

        if row == c:

            correctPrediction +=cm[row,c]

        else:

            falsePrediction += cm[row,c]

print('Correct predictions: ', correctPrediction)

print('False predictions', falsePrediction)

print ('\n\nAccuracy of the Naive Bayes Clasification is: ', correctPrediction/(cm.sum()))
from sklearn import metrics

print(metrics.classification_report(y_pred, y_test))
#Import library

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
diabetes = pd.read_csv('pima-indians-diabetes.csv')

print(diabetes.columns)
# Eye ball the imported dataset

diabetes.head()
print("dimension of diabetes data: {}".format(diabetes.shape))

#The diabetes dataset consists of 768 data points, with 9 features
print(diabetes.groupby('class').size())
import seaborn as sns



sns.countplot(diabetes['class'],label="Count")
diabetes.info()
colormap = plt.cm.viridis # Color range to be used in heatmap

plt.figure(figsize=(15,15))

plt.title('Pearson Correlation of attributes', y=1.05, size=19)

sns.heatmap(diabetes.corr(),linewidths=0.1,vmax=1.0, 

            square=True, cmap=colormap, linecolor='white', annot=True)

#There is no strong correlation between any two variables.

#There is no strong correlation between any independent variable and class variable.
spd = pd.plotting.scatter_matrix(diabetes, figsize=(20,20), diagonal="kde")
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'class'], diabetes['class'], stratify=diabetes['class'], random_state=11)
X_train.shape
from sklearn.svm import SVC



svc = SVC()

svc.fit(X_train, y_train)



print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))

print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))
#The model overfits substantially with a perfect score on the training set and only 65% accuracy on the test set.



#SVM requires all the features to be on a similar scale. We will need to rescale our data that all the features are approximately on the same scale and than see the performance
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)
svc = SVC()

svc.fit(X_train_scaled, y_train)



print("Accuracy on training set: {:.2f}".format(svc.score(X_train_scaled, y_train)))

print("Accuracy on test set: {:.2f}".format(svc.score(X_test_scaled, y_test)))
svc = SVC(C=1000)

svc.fit(X_train_scaled, y_train)



print("Accuracy on training set: {:.3f}".format(

    svc.score(X_train_scaled, y_train)))

print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))