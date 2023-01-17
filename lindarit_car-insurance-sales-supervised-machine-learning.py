# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns



from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.utils import resample

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn import neighbors

from sklearn.metrics import confusion_matrix, precision_score, recall_score
data = pd.read_csv("/kaggle/input/ibm-watson-marketing-customer-value-data/WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv")
data.head()
data.columns
data.shape
data.dtypes
data.describe()
sns.countplot("Response", hue="Gender", data = data)
data.Response.value_counts()
print("Only",round((len(data[(data.Response == "Yes")])/len(data.Response)*100),2),"%","of our customer accept an offer made by your Sales Team.")
data.groupby("Sales Channel").agg({"Response":"count"})
channel = list(data["Sales Channel"].unique())

for i in channel:

    output = len(data[(data["Sales Channel"] == i) & 

                      (data["Response"] == "Yes")]) /len(data[(data["Sales Channel"] == i)])

    print(round((output * 100),2), "% of offers via the Sales Channel", i, "were accepted.")
data.dtypes
objects = ["State","Response","Coverage","Education","EmploymentStatus",

           "Gender","Location Code","Marital Status","Policy Type","Policy","Renew Offer Type","Sales Channel",

           "Vehicle Class","Vehicle Size"]



for obj in objects:

    print(data[obj].value_counts())
data = data.drop(columns={"Customer","Policy", "Effective To Date"})
# Define a list with all features which are categorial



data_categorial = data.select_dtypes(include=["object"])

categories = list(data_categorial.columns)

categories
# Encode the categorial Data to numerical



lb = LabelEncoder()



for i in categories:

    data[i] = lb.fit_transform(data[i])

sns.heatmap(data.corr())
y = data["Response"]
X = data.drop(["Response"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=29)
lr = LogisticRegression()

# initialize the model (=lr)



lr.fit(X_train,y_train)

#fit the model to the train set



acc = lr.score(X_test,y_test)*100

# comapring the test with the data



print("Logistic Regression Test Accuracy", round(acc, 2),"%")
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k

knn.fit(X_train, y_train)

# prediction = knn.predict(x_test)



y_pred = knn.predict(X_test)



acc = knn.score(X_test, y_test)*100

print("2 neighbors KNN Score: ",round(acc,2),"%")
from sklearn.svm import SVC

svm = SVC()

svm.fit(X_train, y_train)



acc = svm.score(X_test,y_test)*100

print("SVM Algorithm Test Accuracy", round(acc, 2),"%")
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)



acc = dtc.score(X_test, y_test)*100

print("Decision Tree Test Accuracy", round(acc, 2),"%")
#Downsampling:



#1. Test-Train Split!!

# concatenate our training data back together



X_down = pd.concat([X_train, y_train], axis=1)



# separate minority and majority classes



no_effect = X_down[X_down.Response==0]

effect = X_down[X_down.Response==1]



# downsample majority



no_effect_downsampled = resample(no_effect,

                               replace = False, # sample without replacement

                               n_samples = len(effect), # match minority n

                               random_state = 27) # reproducible results



# combine minority and downsampled majority



downsampled = pd.concat([no_effect_downsampled, effect])



# checking counts



downsampled.Response.value_counts()
downsampled.shape
y_train_down = downsampled.Response
X_train_down = downsampled.drop(["Response"], axis = 1)
lr = LogisticRegression()

# initialize the model (=lr)



lr.fit(X_train_down,y_train_down)

#fit the model to the train set



y_pred = lr.predict(X_test)



acc = lr.score(X_test,y_test)*100

# comapring the test with the data



print("Prediction",y_pred[:5])

print("Logistic Regression Test Accuracy", round(acc, 2),"%")
n_neighbors = 2

knn = KNeighborsClassifier(n_neighbors = n_neighbors)  # n_neighbors means k

knn.fit(X_train_down, y_train_down)



y_pred = knn.predict(X_test)



acc = knn.score(X_test, y_test)*100



print("Prediction:", y_pred[:5])

print(n_neighbors,"neighbors KNN Score: ",round(acc,2),"%")
acc_train = knn.score(X_train, y_train)*100

print("The accuracy score for the training data is: ",round(acc_train,2),"%")

acc_test = knn.score(X_test,y_test)*100

print("The accuracy score for the test data is: ",round(acc_test,2),"%")



cv_results = cross_val_score(knn, X_train_down,y_train_down, cv = 5)

cv_results
dtc = DecisionTreeClassifier()

dtc.fit(X_train_down, y_train_down)



y_pred_dtc = dtc.predict(X_test)



acc_dtc = dtc.score(X_test, y_test)*100



print("Prediction", y_pred_dtc[:5])

print("Decision Tree Test Accuracy", round(acc_dtc, 2),"%")
acc_train = dtc.score(X_train, y_train)*100

print("The accuracy score for the training data is: ",round(acc_train,2),"%")

acc_test = dtc.score(X_test,y_test)*100

print("The accuracy score for the test data is: ",round(acc_test,2),"%")
cv_results = cross_val_score(dtc, X_train_down,y_train_down, cv = 5)

cv_results
cnf_matrix = confusion_matrix(y_test, y_pred_dtc)

cnf_matrix
dtc_recall = recall_score(y_test, y_pred_dtc)

dtc_recall
271/(271+4)
dtc_precision = precision_score(y_test,y_pred_dtc)

dtc_precision