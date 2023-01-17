# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import csv

import os



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load Titanic dataset

train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
train.head(30)
print('Size of Training Data: ', train.shape)
# Extract titles from names and add to dataset

def extractTitle(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'Unknown'



# Creates a list with the all possible titles

titles = sorted(set([x for x in train.Name.map(lambda x: extractTitle(x))]))

print('Different titles found on the dataset:')

print(len(titles), ':', titles)

print()



# Normalize titles, returns 0, 1, 2, 3, or 4.

def replaceTitle(x):

    title = x['Title']

    

    if title in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir', 'Mr']:

        return 1

    elif title in ['the Countess', 'Mme', 'Lady', 'Mrs']:

        return 2

    elif title in ['Master']:

        return 3

    elif title in ['Mlle', 'Ms', 'Miss']:

        return 4

    elif title =='Dr':

        if x['Sex']=='male':

            return 1

        else:

            return 2

    else:

        return 0



# Creating a new column for the title and replace with the numbered code

train['Title'] = train['Name'].map(lambda x: extractTitle(x))

train['Title'] = train.apply(replaceTitle, axis=1)



# Plot results

train.Title.value_counts().plot(kind='bar')

plt.xlabel('Title Groups', fontsize=15)

plt.ylabel('Number of Passengers', fontsize=15)

plt.title('Titles of Passengers', fontsize=15)



# Extract Titles from test data

test['Title'] = test['Name'].map(lambda x: extractTitle(x))

test['Title'] = test.apply(replaceTitle, axis=1)
train.head(5)
# Displays the portion of the dataset the is nonexistant, by feature

def missingData(data):

    print('Information on Data:\n')

    data.info()

    print('\nShow amount of missing data for each feature:\n')

    tot = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    md=pd.concat([tot, percent], axis=1, keys=['Total', 'Percent'])

    md= md[md["Percent"] > 0]

    f,ax =plt.subplots(figsize=(8,6))

    plt.xticks(rotation='90')

    fig=sns.barplot(md.index, md["Percent"],color="Red",alpha=0.8)

    plt.xlabel('Features', fontsize=15)

    plt.ylabel('Percent of missing values', fontsize=15)

    plt.title('Percent missing data by feature', fontsize=15)

    return md
missingData(train)
missingData(test)
# Remove the cabin feature

train.drop('Cabin', axis=1, inplace = True)

test.drop('Cabin', axis=1, inplace = True)



train.head(5)
# Fills Age feature with the Median age of its respective Title.

def fillAge(featureData):

    featureData["Age"].fillna(featureData.groupby("Title")["Age"].transform("median"), inplace = True)

    

# Call above function to complete Age feature in training and test dataset

fillAge(train)

fillAge(test)



# Display training dataset

train.head(10)
# Filling in NaN data in train

train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)



train.head(5)
# fare.py

# filling in missing fares



ClassA = train['Pclass']

FareA = train['Fare']



# Convert values to numeric, skipping header row

ClassA = np.array([int(i) for i in ClassA[1:]])

FareA = np.array([float(i) for i in FareA[1:]])



# Get Mean for each class

meanFareD = dict()

for i in np.unique(ClassA):

    meanFareD[i] = np.mean(FareA[np.where(

        np.logical_and(ClassA==i, FareA!=0.0))])



print("There are " + str(len(np.where(FareA == 0.0)[0])) + " fares of 0")

[print("The mean for class " + str(i) + " is " + str(meanFareD[i]))

    for i in meanFareD.keys()]



# replace zero fares with average fare of class

FareA = [FareA[i] if FareA[i]!=0.0 else meanFareD[ClassA[i]]

    for i in range(len(FareA))]

print("There are " + str(len(np.where(FareA == 0.0)[0])) + " fares of 0")
# Fills Fare feature with the mean Fare of its respective Pclass.

def fillFare(featureData):

    featureData["Fare"].fillna(featureData.groupby("Pclass")["Fare"].transform("mean"), inplace = True)

    

# Call above function to complete Fare feature in test dataset

fillFare(test)
# Map gender to 0 (male) or 1 (female)

train['Sex'] = train['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

test['Sex']  = test['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
# Converting Embarked feature from text to numeric data

train['Embarked'] = train['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)

test['Embarked'] = test['Embarked'].map( {'Q': 0, 'S': 1, 'C': 2} ).astype(int)
train.head(5)
from statistics import median, mean



# Print mean and median of fare data (to get a better idea for the criteria of the groups)

print('Mean of Fare:', mean(train['Fare']))

print('Median of Fare:', median(train['Fare']))



# Groups fare data into 5 groups

def groupFareData(featureData):

    for dataset in featureData:

        dataset.loc[(dataset['Fare'] <= 15), 'Fare'] = 0,                           # Group 0

        dataset.loc[(dataset['Fare'] > 15) & (dataset['Fare'] <= 30), 'Fare'] = 1,  # Group 1    

        dataset.loc[(dataset['Fare'] > 30) & (dataset['Fare'] <= 60), 'Fare'] = 2,  # Group 2

        dataset.loc[(dataset['Fare'] > 60) & (dataset['Fare'] <= 100), 'Fare'] = 3, # Group 3

        dataset.loc[(dataset['Fare'] > 100), 'Fare'] = 4                            # Group 4



# Call above function to group Fare data in training and test dataset

groupFareData([train, test])



# Display number of passengers in each group

print('\nNumber of Passengers in each Fare group:')

print(train['Fare'].value_counts())



# Print data to see if Fare data has been changed to be in the correct group

train.head(10)
# Groups age data into 5 groups

def groupAgeData(featureData):

    for dataset in featureData:

        dataset.loc[(dataset['Age'] <= 16), 'Age'] = 0,                         # Group 0

        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 25), 'Age'] = 1, # Group 1    

        dataset.loc[(dataset['Age'] > 25) & (dataset['Age'] <= 40), 'Age'] = 2, # Group 2

        dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 62), 'Age'] = 3, # Group 3

        dataset.loc[(dataset['Age'] > 62), 'Age'] = 4                           # Group 4

        

# Call above function to group Age data in training and test dataset

groupAgeData([train, test])



# Display number of passengers in each group

print('\nNumber of Passengers in each Age group:')

print(train['Age'].value_counts())



# Print data to see if Age data has been changed to be in the correct group

train.head(10)
# Extracts PersonType from Age and Sex feature

def extractPersonType(featureData):

        for dataset in featureData:

            dataset.loc[dataset['Age'] == 0, 'PersonType'] = 0, # If passenger is a child

            dataset.loc[(dataset['Age'] != 0) & (dataset['Sex'] == 1), 'PersonType'] = 1, # Passenger is a Woman

            dataset.loc[(dataset['Age'] != 0) & (dataset['Sex'] == 0), 'PersonType'] = 2 # Passenger is a Man



# Extract feature in training data

extractPersonType([train])

train.head(5)



# Extract feature in test data

extractPersonType([test])
# Extract FamilySize from SibSp and Parch feature

def extractFamilySize(featureData):

    for dataset in featureData:

        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch']



# Extract feature in training data

extractFamilySize([train])

train.head(10)



# Extract feature in test data

extractFamilySize([test])
# Plots a graph showing passengers that survived vs dead for each category of the feature

# Using original training dataset (before data conditioning)



def plotOrigFeature(feature, xlabel, title):

    

    # Load initial training data

    train = pd.read_csv('/kaggle/input/titanic/train.csv')

    

    # Get data on feature

    SurvivedA = np.array(train['Survived'])

    FeatureA = np.array(train[feature])

    

    # If feature is Ticket Number

    if (feature == 'Ticket'):

        # Split string part of Ticket Number

        TicketNumA = np.array([i.split()[-1] for i in FeatureA[1:]])

        TicketStrA = np.array([i.split()[0] if len(i.split()) > 1 else '' for i in FeatureA[1:]])

        

        # Save keep only alphanumeric characters

        FeatureA = np.array([''.join(ch for ch in s.upper() if ch.isalnum()) for s in TicketStrA])

        

        print("Unique TicketNum = " + str(len(np.unique(TicketNumA))))

        print("Unique TicketStr = " + str(len(np.unique(FeatureA))))

   

    # Compute expected value of survival given feature data

    FeatureProb = np.array([np.mean(SurvivedA[np.where(FeatureA==i)]) for i in np.unique(FeatureA)])

    

    # Compute percentage of survived / died for each feature value

    FeaturePctSurv = FeatureProb*np.array(

        [np.size(arr) for arr in [np.where(FeatureA==i) for i in np.unique(FeatureA)]], dtype=np.single

        )/len(FeatureA)

    FeaturePctDied = (1-FeatureProb)*np.array(

        [np.size(arr) for arr in [np.where(FeatureA==i) for i in np.unique(FeatureA)]], dtype=np.single

        )/len(FeatureA)

    

    # Plot results

    labels = np.unique(FeatureA)

    fig,ax = plt.subplots()

    x = np.arange(len(labels))

    w = 0.5



    ax.bar(x-w/2, FeaturePctDied, w, label='Died', color='red', bottom=FeaturePctSurv)

    ax.bar(x-w/2, FeaturePctSurv, w, label='Survived', color='green')

    ax.set_xticks(x)

    ax.set_xticklabels(labels)    

    ax.set_xlim(-w, len(labels)+3*w)

    ax.set_xlabel(xlabel)

    ax.set_ylim(0, 1.0)

    ax.set_ylabel('Percentages of Passengers')

    ax.set_title(title)

    plt.legend(loc='upper right')

    

    if (feature == 'Ticket'):

        for tick in ax.xaxis.get_major_ticks()[1::2]:

            tick.set_pad(15)

            

    plt.show()

# Plots a graph showing passengers that survived vs dead for each category of feature

# Using training dataset after feature engineering



def plotUpdatedFeature(train, feature, xlabel, title):

        

    # Get data on feature

    SurvivedA = np.array(train['Survived'])

    FeatureA = np.array(train[feature])    

    

    # Compute expected value of survival given feature data

    FeatureProb = np.array([np.mean(SurvivedA[np.where(FeatureA==i)]) for i in np.unique(FeatureA)])

    

    # Compute percentage of survived / died for each feature value

    FeaturePctSurv = FeatureProb*np.array(

        [np.size(arr) for arr in [np.where(FeatureA==i) for i in np.unique(FeatureA)]], dtype=np.single

        )/len(FeatureA)

    FeaturePctDied = (1-FeatureProb)*np.array(

        [np.size(arr) for arr in [np.where(FeatureA==i) for i in np.unique(FeatureA)]], dtype=np.single

        )/len(FeatureA)

    

    # Plot results

    labels = np.unique(FeatureA)

    fig,ax = plt.subplots()

    x = np.arange(len(labels))

    w = 0.5



    ax.bar(x-w/2, FeaturePctDied, w, label='Died', color='red', bottom=FeaturePctSurv)

    ax.bar(x-w/2, FeaturePctSurv, w, label='Survived', color='green')

    ax.set_xticks(x)

    ax.set_xticklabels(labels)    

    ax.set_xlim(-w, len(labels)+3*w)

    ax.set_xlabel(xlabel)

    ax.set_ylim(0, 1.0)

    ax.set_ylabel('Percentages of Passengers')

    ax.set_title(title)

    plt.legend(loc='upper right')

            

    plt.show()

plotUpdatedFeature(train, 'Title', 'Title', 'E(Survival | Title of Passenger)')

print('Passenger count on each Title:\n')

print(train['Title'].value_counts())
plotUpdatedFeature(train, 'Age', 'Age Group', 'E(Survival | Age Group of Passenger)')

print('Passenger count in each Age group:\n')

print(train['Age'].value_counts())
# Call the above plot function

plotOrigFeature('Sex', 'Sex', 'E(Survival | Sex of Passenger)')

print('Passenger count on each Gender Type:\n')

print(train['Sex'].value_counts())
plotUpdatedFeature(train, 'PersonType', 'Person Type', 'E(Survival | Type of Passenger)')

print('Passenger count on each Person Type:\n')

print(train['PersonType'].value_counts())
plotOrigFeature('Pclass', 'Pclass', 'E(Survival | Pclass of Passenger)')

print('Passenger count on each Pclass category:\n')

print(train['Pclass'].value_counts())
plotOrigFeature('SibSp', 'Siblings, Sponses', 'E(Survival | Number of Siblings or Spouses)')

print('Passenger count on each SibSp number:\n')

print(train['SibSp'].value_counts())
plotOrigFeature('Parch', 'Parents, Children', 'E(Survival | Number of Parents or Children)')

print('Passenger count on each Parch number:\n')

print(train['Parch'].value_counts())
plotUpdatedFeature(train, 'FamilySize', 'Family Size', 'E(Survival | Family Size)')

print('Passenger count on each FamilySize number:\n')

print(train['FamilySize'].value_counts())
plotOrigFeature('Ticket', 'Ticket Prefix', 'E(Survival | Ticket Prefix)')
plotUpdatedFeature(train, 'Fare', 'Fare Bin', 'E(Survival | Fare)')

print('Passenger count in each Fare Bin:\n')

print(train['Fare'].value_counts())
plotUpdatedFeature(train, 'Embarked', 'Embarked Location', 'E(Survival | Embarked Location)')

print('Passenger count for each Embarked Location:\n')

print(train['Embarked'].value_counts())
# Save PassengerIds

train_id = train['PassengerId']

test_id = test['PassengerId']



# Remove the PassengerId feature

train.drop('PassengerId', axis=1, inplace = True)

test.drop('PassengerId', axis=1, inplace = True)
# Remove the Name feature

train.drop('Name', axis=1, inplace = True)

test.drop('Name', axis=1, inplace = True)
# Remove the Ticket feature

train.drop('Ticket', axis=1, inplace = True)

test.drop('Ticket', axis=1, inplace = True)
# Import Classifiers

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.svm import SVC



# Import Confusion Matrix

from sklearn.metrics import plot_confusion_matrix

from sklearn.metrics import confusion_matrix

from sklearn.metrics import ConfusionMatrixDisplay
# Separate Training data to fit into classifiers

tr_label = train['Survived'] # Ground truth data

tr_data = train.loc[:,train.columns!='Survived'] # Feature data



# Print final training data

tr_data.head(10)
# Print test data after data conditioning and feature extraction

test.head(15)
# 5-fold Cross Validation

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
# Fit Logistic Regression classier on training data

# logreg = LogisticRegression()

logreg = GridSearchCV(estimator=LogisticRegression(), param_grid={'solver': ['liblinear'],'C': [0.1, 1, 10, 100, 1000], 'penalty': ['l1','l2',]}, cv=5)

logreg.fit(tr_data, tr_label)



# Predict with training data

tr_y_lg = logreg.predict(tr_data).astype(int)



# Get Cross Validation score on classifier

cr_score = cross_val_score(logreg, tr_data, tr_label, cv=k_fold, n_jobs=1, scoring='accuracy')



# Print results

print('Cross Validation score for each iteration (in %): ', cr_score*100)

print('Average CV score: ', round(np.mean(cr_score)*100, 2), '%')



print('Ground Truth Data  =', tr_label[:35].values)

print('Prediction Results =', tr_y_lg[:35])

print("Percentage of Training Data classified correctly:    {:.2f}%".format(np.sum(tr_y_lg==tr_label)/tr_label.size*100))

print("Percentage of Deaths that were predicted to survive: {:.2f}%".format(np.sum((tr_y_lg!=tr_label) & (tr_label==0))/tr_label.size*100))

print("Percentage of Survivals that were predicted to die:  {:.2f}%".format(np.sum((tr_y_lg!=tr_label) & (tr_label==1))/tr_label.size*100))





print()

print("Best Parameters:")

print(logreg.best_params_)

print()

means = logreg.cv_results_['mean_test_score']

stds = logreg.cv_results_['std_test_score']

for mean,std,params in zip(means,stds,logreg.cv_results_['params']):

    print("%0.3f (+/-%0.3f) for %r" % (mean, std*2, params))

    

print()

print(logreg.get_params())
# Plot Confusion Matrix for Logistic Regression classifier

label_names = ['Dead', 'Survived']

disp = plot_confusion_matrix(logreg, tr_data, tr_label, display_labels=label_names,cmap = plt.cm.Blues, normalize='true')

disp.ax_.set_title('Logistic Regression')



print('Normalized Confusion Matrix for Logistic Regression:')

print(disp.confusion_matrix)



plt.show()
# Fit Gaussian Naive Bayes classier on training data

gnb = GaussianNB()

gnb.fit(tr_data, tr_label)



# Predict with training data

tr_y_gnb = gnb.predict(tr_data).astype(int)



# Get Cross Validation score on classifier

cr_score = cross_val_score(gnb, tr_data, tr_label, cv=k_fold, n_jobs=1, scoring='accuracy')



# Print results

print('Cross Validation score for each iteration (in %): ', cr_score*100)

print('Average CV score: ', round(np.mean(cr_score)*100, 2), '%')



print('Ground Truth Data  =', tr_label[:35].values)

print('Prediction Results =', tr_y_gnb[:35])

print("Percentage of Training Data classified correctly:    {:.2f}%".format(np.sum(tr_y_gnb==tr_label)/tr_label.size*100))

print("Percentage of Deaths that were predicted to survive: {:.2f}%".format(np.sum((tr_y_gnb!=tr_label) & (tr_label==0))/tr_label.size*100))

print("Percentage of Survivals that were predicted to die:  {:.2f}%".format(np.sum((tr_y_gnb!=tr_label) & (tr_label==1))/tr_label.size*100))

# Plot Confusion Matrix for Gaussian Naive Bayes classifier

label_names = ['Dead', 'Survived']

disp = plot_confusion_matrix(gnb, tr_data, tr_label, display_labels=label_names,cmap = plt.cm.Blues, normalize='true')

disp.ax_.set_title('Gaussian Naives Bayes')



print('Normalized Confusion Matrix for Gaussian Naives Bayes:')

print(disp.confusion_matrix)



plt.show()
# Fit KNN classifier on training data

knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid={'n_neighbors': range(2,10)}, cv=5)

knn.fit(tr_data, tr_label)



# Predict with training data

tr_y_knn = knn.predict(tr_data).astype(int)



# Get Cross Validation score on classifier

cr_score = cross_val_score(knn, tr_data, tr_label, cv=k_fold, n_jobs=1, scoring='accuracy')



# Print results

print('Cross Validation score for each iteration (in %): ', cr_score*100)

print('Average CV score: ', round(np.mean(cr_score)*100, 2), '%')



print('Ground Truth Data  =', tr_label[:35].values)

print('Prediction Results =', tr_y_knn[:35])

print("Percentage of Training Data classified correctly:    {:.2f}%".format(np.sum(tr_y_knn==tr_label)/tr_label.size*100))

print("Percentage of Deaths that were predicted to survive: {:.2f}%".format(np.sum((tr_y_knn!=tr_label) & (tr_label==0))/tr_label.size*100))

print("Percentage of Survivals that were predicted to die:  {:.2f}%".format(np.sum((tr_y_knn!=tr_label) & (tr_label==1))/tr_label.size*100))



print()

print("Best Parameters:")

print(knn.best_params_)

print()

means = knn.cv_results_['mean_test_score']

stds = knn.cv_results_['std_test_score']

for mean,std,params in zip(means,stds,knn.cv_results_['params']):

    print("%0.3f (+/-%0.3f) for %r" % (mean, std*2, params))

    

print()

print(knn.get_params())
# Plot Confusion Matrix for K-Nearest Neighbor classifier

label_names = ['Dead', 'Survived']

disp = plot_confusion_matrix(knn, tr_data, tr_label, display_labels=label_names,cmap = plt.cm.Blues, normalize='true')

disp.ax_.set_title('K-Nearest Neighbor')



print('Normalized Confusion Matrix for K-Nearest Neighbor:')

print(disp.confusion_matrix)



plt.show()
# Fit Random Forest classier on training data

ranfor = RandomForestClassifier(n_estimators=13)

ranfor.fit(tr_data, tr_label)



# Predict with training data

tr_y_rf = ranfor.predict(tr_data).astype(int)



# Get Cross Validation score on classifier

cr_score = cross_val_score(ranfor, tr_data, tr_label, cv=k_fold, n_jobs=1, scoring='accuracy')



# Print results

print('Cross Validation score for each iteration (in %): ', cr_score*100)

print('Average CV score: ', round(np.mean(cr_score)*100, 2), '%')



print('Ground Truth Data  =', tr_label[:35].values)

print('Prediction Results =', tr_y_rf[:35])

print("Percentage of Training Data classified correctly:    {:.2f}%".format(np.sum(tr_y_rf==tr_label)/tr_label.size*100))

print("Percentage of Deaths that were predicted to survive: {:.2f}%".format(np.sum((tr_y_rf!=tr_label) & (tr_label==0))/tr_label.size*100))

print("Percentage of Survivals that were predicted to die:  {:.2f}%".format(np.sum((tr_y_rf!=tr_label) & (tr_label==1))/tr_label.size*100))

# Plot Confusion Matrix for Random Forest classifier

label_names = ['Dead', 'Survived']

disp = plot_confusion_matrix(ranfor, tr_data, tr_label, display_labels=label_names,cmap = plt.cm.Blues, normalize='true')

disp.ax_.set_title('Random Forest')



print('Normalized Confusion Matrix for Random Forest:')

print(disp.confusion_matrix)



plt.show()
# Fit Perceptron classifier on training data

percep = Perceptron()

percep.fit(tr_data, tr_label)



# Predict with training data

tr_y_p = percep.predict(tr_data).astype(int)



# Get Cross Validation score on classifier

cr_score = cross_val_score(percep, tr_data, tr_label, cv=k_fold, n_jobs=1, scoring='accuracy')



# Print results

print('Cross Validation score for each iteration (in %): ', cr_score*100)

print('Average CV score: ', round(np.mean(cr_score)*100, 2), '%')



print('Ground Truth Data  =', tr_label[:35].values)

print('Prediction Results =', tr_y_p[:35])

print("Percentage of Training Data classified correctly:    {:.2f}%".format(np.sum(tr_y_p==tr_label)/tr_label.size*100))

print("Percentage of Deaths that were predicted to survive: {:.2f}%".format(np.sum((tr_y_p!=tr_label) & (tr_label==0))/tr_label.size*100))

print("Percentage of Survivals that were predicted to die:  {:.2f}%".format(np.sum((tr_y_p!=tr_label) & (tr_label==1))/tr_label.size*100))

# Plot Confusion Matrix for Perceptron classifier

label_names = ['Dead', 'Survived']

disp = plot_confusion_matrix(percep, tr_data, tr_label, display_labels=label_names,cmap = plt.cm.Blues, normalize='true')

disp.ax_.set_title('Perceptron')



print('Normalized Confusion Matrix for Perceptron:')

print(disp.confusion_matrix)



plt.show()
# Fit SVM classier on training data

svm = SVC()

svm.fit(tr_data, tr_label)



# Predict with training data

tr_y_svm = svm.predict(tr_data).astype(int)



# Get Cross Validation score on classifier

cr_score = cross_val_score(svm, tr_data, tr_label, cv=k_fold, n_jobs=1, scoring='accuracy')



# Print results

print('Cross Validation score for each iteration (in %): ', cr_score*100)

print('Average CV score: ', round(np.mean(cr_score)*100, 2), '%')



print('Ground Truth Data  =', tr_label[:35].values)

print('Prediction Results =', tr_y_svm[:35])

print("Percentage of Training Data classified correctly:    {:.2f}%".format(np.sum(tr_y_svm==tr_label)/tr_label.size*100))

print("Percentage of Deaths that were predicted to survive: {:.2f}%".format(np.sum((tr_y_svm!=tr_label) & (tr_label==0))/tr_label.size*100))

print("Percentage of Survivals that were predicted to die:  {:.2f}%".format(np.sum((tr_y_svm!=tr_label) & (tr_label==1))/tr_label.size*100))

# Plot Confusion Matrix for Support Vector Machine classifier

label_names = ['Dead', 'Survived']

disp = plot_confusion_matrix(svm, tr_data, tr_label, display_labels=label_names,cmap = plt.cm.Blues, normalize='true')

disp.ax_.set_title('Support Vector Machine')



print('Normalized Confusion Matrix for SVM:')

print(disp.confusion_matrix)



plt.show()
# Calculate Majority vote for each passenger

def calcMajority(predictions):

    

    # Get number of passengers

    num = predictions.shape[1]

    

    # Create results array

    results = np.empty(num)

    

    # Iterate through each passenger

    i = 0

    for col in predictions.T:

        # Calculate Vote for 1 (Passenger Survived)

        votes = col.sum()

        

        # Check if vote is majority

        if votes >= 3:

            # Majority vote is 1

            results[i] = 1

        else:

            # Majority vote is 0

            results[i] = 0

        

        i = i + 1

        

    return results.astype(int)
# Results for Ensemble model on training data

tr_preds = np.array([tr_y_lg, tr_y_gnb, tr_y_knn, tr_y_rf, tr_y_svm])

tr_results = calcMajority(tr_preds)



# Print results

print('Ground Truth Data  =', tr_label[:35].values)

print('Prediction Results =', tr_results[:35])

print("Percentage of Training Data classified correctly:    {:.2f}%".format(np.sum(tr_results==tr_label)/tr_label.size*100))

print("Percentage of Deaths that were predicted to survive: {:.2f}%".format(np.sum((tr_results!=tr_label) & (tr_label==0))/tr_label.size*100))

print("Percentage of Survivals that were predicted to die:  {:.2f}%".format(np.sum((tr_results!=tr_label) & (tr_label==1))/tr_label.size*100))
# Plot Confusion Matrix (Training Data)

label_names = ['Dead', 'Survived']

cm = confusion_matrix(tr_label, tr_results, normalize = 'true')

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)

disp = disp.plot(cmap=plt.cm.Blues)

disp.ax_.set_title('Ensemble Model - Training')



print('Normalized Confusion Matrix for Ensemble Model:')

print(cm)



plt.show()
# Predict survival of passengers in test data for all the classifiers

pred_lg = logreg.predict(test).astype(int)

pred_gnb = gnb.predict(test).astype(int)

pred_knn = knn.predict(test).astype(int)

pred_rf = ranfor.predict(test).astype(int)

pred_svm = svm.predict(test).astype(int)



# Calculate Majority Vote

test_preds = np.array([pred_lg, pred_gnb, pred_knn, pred_rf, pred_svm])

test_results = calcMajority(test_preds)
# Create Submission file

def createSubmission(predictions):

    submission = pd.DataFrame({

        "PassengerId": test_id,

        "Survived": predictions

    })



    submission.to_csv('submission.csv', index=False)



createSubmission(test_results)



# Print file

submission = pd.read_csv('submission.csv')

submission.head()