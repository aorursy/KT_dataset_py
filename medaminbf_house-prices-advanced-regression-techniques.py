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
#0. Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



#1. Load the data

train = pd.read_csv("train.csv") #Load train data

test = pd.read_csv("test.csv") #Load test data

data = train.append(test,sort=False) #Make train set and test set in the same data set



#2. Clean the data

    #2.1 Dealing with NULL values



        #Plot features with more than 1000 NULL values

features = []

nullValues = []

for i in data:

    if (data.isna().sum()[i])>1000 and i!='SalePrice':

        features.append(i)

        nullValues.append(data.isna().sum()[i])

y_pos = np.arange(len(features))

plt.bar(y_pos, nullValues, align='center', alpha=0.5)

plt.xticks(y_pos, features)

plt.ylabel('NULL Values')

plt.xlabel('Features')

plt.title('Features with more than 1000 NULL values')

plt.show()



        #Dealing with NULL values

            #Drop columns that contain more than 1000 NULL values

data = data.dropna(axis=1, how='any', thresh = 1000) #thresh=1000 : Keep the rows with at least 1000 non-NA values

            #Replace NULL values with mean values

data = data.fillna(data.mean())

    #2.2 Dealing with string values

        #Convert string values to integer values

data = pd.get_dummies(data)

    #2.3 Dealing with correlations : Correlation describes the association between random variables

        #Drop features that are correlated to each other

covarianceMatrix = data.corr()

listOfFeatures = [i for i in covarianceMatrix]

setOfDroppedFeatures = set()

for i in range(len(listOfFeatures)) :

    for j in range(i+1,len(listOfFeatures)): #Avoid repetitions

        feature1=listOfFeatures[i]

        feature2=listOfFeatures[j]

        if abs(covarianceMatrix[feature1][feature2]) > 0.8: #If the correlation between the features is > 0.8 : 0.8 was the one that gave the best results for me

            setOfDroppedFeatures.add(feature1) #Add one of them to the set

data = data.drop(setOfDroppedFeatures, axis=1) #Drop correlated features

        #Drop features that are not correlated with output

nonCorrelatedWithOutput = [column for column in data if abs(data[column].corr(data["SalePrice"])) < 0.05] #0.045 was the one that gave the best results for me

data = data.drop(nonCorrelatedWithOutput, axis=1)

    #2.4 Dealing with outliers : outlier is an observation point that is distant from other observations

        # First:seperating the data (Because removing outliers ⇔ removing rows, and we don't want to remove rows from test set)

newTrain = data.iloc[:1460]

newTest = data.iloc[1460:]

        # Second:defining a function that returns outlier values using percentile() method

def outliers_iqr(ys):

    quartile_1, quartile_3 = np.percentile(ys, [25, 75])  # Get 1st and 3rd quartiles (25% -> 75% of data will be kept)

    iqr = quartile_3 - quartile_1

    lower_bound = quartile_1 - (iqr * 1.5)  # Get lower bound

    upper_bound = quartile_3 + (iqr * 1.5)  # Get upper bound

    return np.where((ys > upper_bound) | (ys < lower_bound))  # Get outlier values

        # Third:droping the outlier values from the train set

trainWithoutOutliers = newTrain  # We can't change train while running through it

for column in newTrain:

    outlierValuesList = np.ndarray.tolist(outliers_iqr(newTrain[column])[0])  # outliers_iqr() returns an array

    trainWithoutOutliers = newTrain.drop(outlierValuesList)  # Drop outlier rows

trainWithoutOutliers = newTrain



#3. Train the data

    # Importing Classifier Modules

from sklearn.linear_model import LinearRegression           # LinearRegression

from sklearn.neighbors import KNeighborsClassifier          #KNeighbors

from sklearn.tree import DecisionTreeClassifier             #DecisionTree

from sklearn.ensemble import RandomForestClassifier         #RandomForest

from sklearn.naive_bayes import GaussianNB                  #GaussianNB

from sklearn.svm import SVC                                 #SVC

from sklearn.model_selection import KFold                   #KFold

from sklearn.model_selection import cross_val_score         #cross_val_score

k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

    #Training and scoring

train_data = trainWithoutOutliers.drop("SalePrice", axis=1) #Remove SalePrice column

target = np.log1p(trainWithoutOutliers["SalePrice"]) #Get SalePrice column {log1p(x) = log(x+1)}

       #kNN

"""

print("kNN")

clf = KNeighborsClassifier(n_neighbors = 13)

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(round(np.mean(score)*100, 2))

        #Decision Tree

print("Decision Tree")

clf = DecisionTreeClassifier()

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(round(np.mean(score)*100, 2))

        #Ramdom Forest

print("Ramdom Forest")

for i in range (20) :

    clf = RandomForestClassifier(n_estimators=i+1)

    scoring = 'accuracy'

    score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

    print(i , round(np.mean(score)*100, 2))

        #Naive Bayes

print("Naive Bayes")

clf = GaussianNB()

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(round(np.mean(score)*100, 2))

        #SVM

print("SVC ")

clf = SVC()

scoring = 'accuracy'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(round(np.mean(score)*100, 2)) """

        #LinearRegression

print("LinearRegression ")

clf = LinearRegression()

scoring = 'neg_mean_squared_error'

score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)

print(round(np.mean(score)*100, 2))



#4. Make & Submit prediction

    #4.1 Fitting the Classifier

clf = LinearRegression()

clf.fit(train_data, target)

    #4.2 Make prediction

newTest = newTest.drop("SalePrice", axis=1) #Remove SalePrice column

prediction = np.expm1(clf.predict(newTest))

#prediction = np.expm1(reg.predict(newTest)) #{expm1(x) = exp(x)-1}

    #4.3 Submit prediction

submission = pd.DataFrame() #Create a new DataFrame for submission

submission['Id'] = test['Id']

submission['SalePrice'] = prediction

submission.to_csv("submissionLR2.csv", index=False) #Convert DataFrame to .csv file

submission = pd.read_csv("submissionLR2.csv")








