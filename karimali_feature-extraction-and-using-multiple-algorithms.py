# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Getting the data

df = pd.read_csv("../input/data.csv")

df.head(20)
df.shape
df.info()
#As Unnamed: 32 has all the NULL values so it will not be useful for our model

df.drop("Unnamed: 32",axis=1,inplace=True)
#ID will also be not required for the model as each value is unique 

df.drop("id",axis=1,inplace=True)
df.head()
#Lets count how many people are on which stage

sns.countplot(df['diagnosis'],label="Count")
#So according to the description we are dividing the features into 3 categories

mean_features = list(df.columns[1:10])

se_features = list(df.columns[10:20])

worst_features = list(df.columns[20:30]) 
print(mean_features)

print("****************************************************")

print(se_features)

print("****************************************************")

print(worst_features)
#Finding the similarities between mean_features

corr = df[mean_features].corr()

plt.figure(figsize=(15,15))

sns.heatmap(corr, cbar = True,  square = True, annot=True, fmt= '.2f',annot_kws={'size': 15},

           cmap= 'coolwarm')
#So with the help of our heat map we know that:

#Compactness_mean, concavity_mean and concavepoint_mean are highly correlated

#Radius,area and parimeter are highly correlated

#So we will select one from the correlated features
#So our selected final features are Compactness_mean,Radius_mean,texture_mean,smoothness_mean,symmetry_mean
pred_features = ['radius_mean','compactness_mean','texture_mean','smoothness_mean','symmetry_mean']
X = df[pred_features]

y = df.diagnosis
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape)

print(y_test.shape)
#Applying "feature scaling"

#Feature scaling is a method used to standardize the range of independent variables or features of data

from sklearn.preprocessing import StandardScaler as SS

sc_X = SS()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

classifier.fit(X_train,y_train)
ypred=classifier.predict(X_test)

rf = metrics.accuracy_score(ypred,y_test) * 100 # to check the accuracy

print(rf)
#So we got an accuracy of 94.73% if we use our selected mean features and Random forest

#Now let us try some different models for classification
#Let us apply SVM model

from sklearn.svm import SVC

classifier = SVC(kernel ='linear',random_state=0)

classifier.fit(X_train,y_train)
ypred=classifier.predict(X_test)

svm = metrics.accuracy_score(ypred,y_test) * 100 # to check the accuracy

print(svm)
#Now let us apply decision tree classifier

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)

classifier.fit(X_train,y_train)
ypred=classifier.predict(X_test)

dtc = metrics.accuracy_score(ypred,y_test) * 100 # to check the accuracy

print(dtc)
#Now let us apply K_NN classifier

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors = 30,metric = 'minkowski',p=2)

classifier.fit(X_train,y_train)
ypred=classifier.predict(X_test)

knn = metrics.accuracy_score(ypred,y_test) *100 # to check the accuracy

print(knn)
#Let us apply Naive Bayes

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train,y_train)
ypred=classifier.predict(X_test)

nb=metrics.accuracy_score(ypred,y_test) * 100 # to check the accuracy

print(nb)
#let us apply Logistic Eegression

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state=0)

classifier.fit(X_train,y_train)
ypred=classifier.predict(X_test)

lr = metrics.accuracy_score(ypred,y_test) * 100 # to check the accuracy

print(lr)
#Comparring results from different Algorithms

objects = ('Random Forest', 'K-NN', 'SVM', 'Naive Bayes', 'Logistic Regression', 'Decision Tree Classifier')

y_pos = np.arange(len(objects))

performance = [rf,knn,svm,nb,lr,dtc]



plt.scatter(y_pos, performance, alpha=1)

plt.plot(y_pos, performance,color='blue')



plt.xticks(y_pos, objects)

plt.ylabel('Accuracy %')

plt.xticks(rotation=45)

plt.title('Algorithm Accuracy')

plt.show()