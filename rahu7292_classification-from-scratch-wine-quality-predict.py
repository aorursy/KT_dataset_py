# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))

print()

print("The files in the dataset are:-")

from subprocess import check_output

print(check_output(['ls','../input']).decode('utf'))



# Any results you write to the current directory are saved as output.







# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



df = pd.read_csv('../input/winequality-red.csv')
df.head()
df.shape
df.info()
# Let us see the correlation between the different variables

# We will reduce the number of features from Principle Componemt Analysis (PCA)

plt.figure(figsize=(12,6))

sns.heatmap(data=df.corr())

plt.show()
# Let us Import the Important Libraries  to train our Model for Machine Learning 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler      # For Scaling the dataset

from sklearn.model_selection import cross_val_score   # To check the accuracy of the model.
# Converting the DataFrame into Feature matrix and Target Vector.

x_train = df.iloc[:,:-1].values

y_train = df.iloc[:,-1].values
# Let us use scaling on our dataset.



sc_x = StandardScaler()

x_train = sc_x.fit_transform(x_train)

from sklearn.decomposition import PCA

pca = PCA(n_components=None, )

x_train = pca.fit_transform(x_train)

explain_variance = pca.explained_variance_ratio_

explain_variance
# Let us apply PCA.

""" 

pca = PCA(n_components=5)

x_train = pca.fit_transform(x_train)

"""
def all_models():

    # Apply One model at a time , not all in a single function. If we run all in a single function then it will take too much memory(RAM) and time.



    # Apply Logistic regression

    # First step is to train our model .



    classifier_logi = LogisticRegression()

    classifier_logi.fit(x_train,y_train)



    # Let us check the accuracy of the model

    accuracy = cross_val_score(estimator=classifier_logi, X=x_train, y=y_train, cv=10)

    print(f"The accuracy of the Logistic Regressor Model is \t {accuracy.mean()}")

    print(f"The deviation in the accuracy is \t {accuracy.std()}")

    print()







    # Apply SVM with Gaussian kernel

    classifier_svm2 = SVC(kernel='rbf', )

    classifier_svm2.fit(x_train,y_train)

    accuracy = cross_val_score(estimator=classifier_svm2, X=x_train, y=y_train, cv=10)

    print(f"The accuracy of the SVM Gaussian kernel Model is \t {accuracy.mean()}")

    print(f"The deviation in the accuracy is \t {accuracy.std()}")

    print()







    # Apply K_NN Model

    # Train model

    classifier_knn = KNeighborsClassifier()

    classifier_knn.fit(x_train,y_train)

    # Check the accuracy.

    accuracy = cross_val_score(estimator=classifier_knn, X=x_train, y=y_train, cv=10)

    print(f"The accuracy of the KNN Model is \t {accuracy.mean()}") 

    print(f"The deviation in the accuracy is \t {accuracy.std()}")

    print()





    # Apply Naive Bayes Model.

    # Train Model

    classifier_bayes = GaussianNB()

    classifier_bayes.fit(x_train,y_train)

    # Check the accuracy and deviation in the accuracy

    accuracy = cross_val_score(estimator=classifier_bayes, X=x_train, y=y_train, cv=10)

    print(f"The accuracy of the Naive Bayes Model is \t {accuracy.mean()}") 

    print(f"The deviation in the accuracy is \t {accuracy.std()}")

    print()





    # Apply Random Forest Model.

    # Train Model

    classifier_ran = RandomForestClassifier(n_estimators=10, criterion='entropy')

    classifier_ran.fit(x_train,y_train)

    # Check the accuracy and deviation in the accuracy

    accuracy = cross_val_score(estimator=classifier_ran, X=x_train, y=y_train, cv=10)

    print(f"The accuracy of the Random Forest Model is \t {accuracy.mean()}") 

    print(f"The deviation in the accuracy is \t {accuracy.std()}")

    print()

    

    return classifier_svm2
# Let us run the all_models funtion and see the accuracies of all model

classifier = all_models()
# Our Target vector

y_train[:50]
# Making prediction on our Feature matrix and compare it with our Target vector.

classifier.predict(x_train)[:50]