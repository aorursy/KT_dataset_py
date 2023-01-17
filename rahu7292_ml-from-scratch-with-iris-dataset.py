# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('ggplot')

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot, download_plotlyjs

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))

print()

print("The files in the dataset are:-- ")

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# IMporting the dataset

df = pd.read_csv("../input/Iris.csv")
# Checking the top 5 entries

df.head()
print(f"The number of rows and columns in the dataset are \t {df.shape}")
# Let's check the unique Species in the dataset, which we will predict in the end.

print(df['Species'].unique())

print("There are 3 species .")
# Let us check whether we have null values in the dataset or not.

print(df.isnull().sum())

print()

print()

print("As one can see there is No Null Values in the dataset.")
# Let us remove the unwanted columns/features which will not help us to predict the Species of the Flower

df.drop('Id', axis=1, inplace=True)
# Let us see the distribution of the SepalLength,SepalWidth,PetalLength, PetalWidth

# And get the Statistical Knowledge of the dataset

temp_df = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm', 'PetalWidthCm']]

temp_df.iplot(kind='box', title='Distribution of Length and Width of Sepal and Petal in Cm', yTitle='Frequency')
df.corr().iplot(kind='heatmap', )
# Let us Import the Important Libraries  to train our Model for Machine Learning 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder, OneHotEncoder # To deal with Categorical Data in Target Vector.

from sklearn.model_selection import train_test_split  # To Split the dataset into training data and testing data.

from sklearn.model_selection import cross_val_score   # To check the accuracy of the model.
df.head()
# Creating Feature Matric and Target Vector.

X = df.iloc[:,:-1].values

Y = df.iloc[:,-1].values
# Let us check whether we have null values in the dataset or not.

print(df.isnull().sum())

print()

print()

print("As one can see there is No Null Values in the dataset.")
# Now we have Categorical data in our Target vector and we need to convert 

# it into values, So that we can easyly perform Mathmethical operations.



label_y = LabelEncoder()

Y = label_y.fit_transform(Y)
Y
df['Species'].unique()
x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2)
# There is no need of Scaling the features.
# First step is to train our model .



classifier_logi = LogisticRegression()

classifier_logi.fit(x_train,y_train)
# Let's Predict our model on test set.

y_pred = classifier_logi.predict(x_test)
# Let us check the accuracy of the model

accuracy = cross_val_score(estimator=classifier_logi, X=x_train, y=y_train, cv=10)

print(f"The accuracy of the Logistic Regressor Model is \t {accuracy.mean()}")

print(f"The deviation in the accuracy is \t {accuracy.std()}")
# Let us tran model

classifier_svm1 = SVC(kernel='linear')

classifier_svm1.fit(x_train,y_train)
# Let's predict on test dataset.

y_pred = classifier_svm1.predict(x_test)
# Check the accuracy.

accuracy = cross_val_score(estimator=classifier_svm1, X=x_train, y=y_train, cv=10)

print(f"The accuracy of the SVM linear kernel Model is \t {accuracy.mean()}")

print(f"The deviation in the accuracy is \t {accuracy.std()}")
# Train the model

classifier_svm2 = SVC(kernel='rbf', )

classifier_svm2.fit(x_train,y_train)
# Predict on test set.

y_pred = classifier_svm2.predict(x_test)
# Check the accuracy.

accuracy = cross_val_score(estimator=classifier_svm2, X=x_train, y=y_train, cv=10)

print(f"The accuracy of the SVM Gaussian kernel Model is \t {accuracy.mean()}")

print(f"The deviation in the accuracy is \t {accuracy.std()}")
# Train model

classifier_knn = KNeighborsClassifier()

classifier_knn.fit(x_train,y_train)
# predict on test set

y_pred = classifier_knn.predict(x_test)
# Check the accuracy.

accuracy = cross_val_score(estimator=classifier_knn, X=x_train, y=y_train, cv=10)

print(f"The accuracy of the KNN Model is \t {accuracy.mean()}") 

print(f"The deviation in the accuracy is \t {accuracy.std()}")
# Train Model

classifier_bayes = GaussianNB()

classifier_bayes.fit(x_train,y_train)
# Predict on test set.

y_pred = classifier_bayes.predict(x_test)
# Check the accuracy and deviation in the accuracy

accuracy = cross_val_score(estimator=classifier_bayes, X=x_train, y=y_train, cv=10)

print(f"The accuracy of the Naive Bayes Model is \t {accuracy.mean()}") 

print(f"The deviation in the accuracy is \t {accuracy.std()}")
# Train model

classifier_deci = DecisionTreeClassifier()

classifier_deci.fit(x_train,y_train)
# Predict on test set

y_pred = classifier_deci.predict(x_test)
# Check the accuracy and deviation in the accuracy

accuracy = cross_val_score(estimator=classifier_deci, X=x_train, y=y_train, cv=10)

print(f"The accuracy of the Decision Tree Model is \t {accuracy.mean()}") 

print(f"The deviation in the accuracy is \t {accuracy.std()}")
# Train Model

classifier_ran = RandomForestClassifier()

classifier_ran.fit(x_train,y_train)
# Predict on test set.

y_pred = classifier_ran.predict(x_test)
# Check the accuracy and deviation in the accuracy

accuracy = cross_val_score(estimator=classifier_ran, X=x_train, y=y_train, cv=10)

print(f"The accuracy of the Random Forest Model is \t {accuracy.mean()}") 

print(f"The deviation in the accuracy is \t {accuracy.std()}")
# Let's make prediction on new values.

try:

    sepalLength = float(input("Enter Sepal Length:\t"))

    sepalWidth = float(input("Enter Sepal Width:\t"))

    petalLength = float(input("Enter Petal Length:\t"))

    petalWidth = float(input("Enter Petal Width:\t"))



    new_values = [[sepalLength,sepalWidth,petalLength,petalWidth],]  # Making 2-D array.



    species = classifier_svm2.predict(new_values) # Using SVM Gaussian kernel



    if species[0]==0:

        flag = 'Iris-setosa'

    elif species[0]==1:

        flag = 'Iris-versicolor'

    else:

        flag = 'Iris-virginica'



    print()

    print()

    print(f"*** The Species is: \t {flag} ****")    

    

except Exception as e:

    print("Run this code with Python")


