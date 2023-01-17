import numpy as np

import pandas as pd

import re as re

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn import preprocessing

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel



%matplotlib inline



# Define working directory

#working_directory = 'D:/Kaggle/1. Titanic - Machine Learning from Disaster/'
########### Load Data ##########  



# Method to load data from csv file

def load_data():

    #train = pd.read_csv("".join((working_directory, 'data/train.csv')))

    #test = pd.read_csv("".join((working_directory, 'data/test.csv')))

    train = pd.read_csv('../input/train.csv')

    test = pd.read_csv('../input/test.csv')

    return train, test

                                       

train, test = load_data()

train.head()
### Title ###



# Get the titles from the names

train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



# Replace the rare titles with a more common title or assign a new title "Officer" or "Royal"

full = [train, test]

for dataset in full:

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev'], 'Officer')

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Sir', 'Jonkheer', 'Dona'], 'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



# Get the average survival rate of different titles

train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
### Age ###



# Get the mean and standard deviation of the ages group by title

title_age = pd.DataFrame(np.concatenate((train[['Title', 'Age']], test[['Title', 'Age']]))) # Concatenate train and test data

title_age.columns = ['Title', 'Age']

title_age = title_age.dropna(axis = 0)

title_age['Age'] = title_age['Age'].astype(int)

# Calculate the mean and standard deviation

avg_age = title_age[['Title', 'Age']].groupby('Title', as_index=False).mean()

std_age = title_age[['Title', 'Age']].groupby('Title', as_index=False)['Age'].apply(lambda x : x.std())

avg_std_age = pd.concat([avg_age, std_age], axis=1)

avg_std_age.columns = ['Title', 'Age', 'Std']

# Calculate the one sigma boundary ((mean - 1 std) and (mean + 1 std))

avg_std_age['Low'] = avg_std_age['Age'] - avg_std_age['Std']

avg_std_age['High'] = avg_std_age['Age'] + avg_std_age['Std']



# Fill missing ages using random ages within 1 standard deviation boundary of different titles

for index, row in avg_std_age.iterrows():

    count_nan_train = train["Age"][train['Title'] == row['Title']].isnull().sum()

    count_nan_test = test["Age"][test['Title'] == row['Title']].isnull().sum()

    train.loc[(np.isnan(train['Age'])) & (train['Title'] == row['Title']), 'Age'] = np.random.randint(row['Low'], row['High'], size = count_nan_train)

    test.loc[(np.isnan(test['Age'])) & (test['Title'] == row['Title']), 'Age'] = np.random.randint(row['Low'], row['High'], size = count_nan_test)



# Convert the data type to integer

train['Age'] = train['Age'].astype(int)

test['Age'] = test['Age'].astype(int)



# Summarise and visualise the total number of passengers by age

total_count = train[['Age', 'Survived']].groupby(['Age'],as_index=False).count()

fig, axis1 = plt.subplots(1,1,figsize=(16,5))

sns.barplot(x='Age', y='Survived', data=total_count, ax=axis1)

axis1.set(xlabel='Age', ylabel='Number of Passenger', title='Number of passenegers by age')



# Summarise and visualise the average survived passengers by age

average_survival = train[['Age', 'Survived']].groupby(['Age'],as_index=False).mean()

fig, axis2 = plt.subplots(1,1,figsize=(16,5))

sns.barplot(x='Age', y='Survived', data=average_survival, ax=axis2)

axis2.set(xlabel='Age', ylabel='Survival Rate', title='Average survival probability by age"')
# Investigate the survival probability after grouping the age into N categories (Here I use N=8)

num_bin = 8

train['AgeBin'] = pd.cut(train['Age'], num_bin)

train[['AgeBin', 'Survived']].groupby(['AgeBin'], as_index=False).mean().sort_values(by='Survived', ascending=True)
# Group the age into N categories and assign integer values to each age category



#max_age = max([train['Age'].max(), test['Age'].max()])

#for i in range(0, num_bin):

#    train.loc[(train['Age'] > max_age / num_bin * i) & (train['Age'] <= max_age / num_bin * (i+1)), 'AgeGroup'] = i

#    test.loc[(test['Age'] > max_age / num_bin * i) & (test['Age'] <= max_age / num_bin * (i+1)), 'AgeGroup'] = i

    

full = [train, test]

for dataset in full:

    dataset.loc[(train['Age'] <= 10), 'AgeGroup'] = 1

    dataset.loc[(train['Age'] > 50) & (train['Age'] <= 60), 'AgeGroup'] = 2

    dataset.loc[(train['Age'] > 30) & (train['Age'] <= 40), 'AgeGroup'] = 3

    dataset.loc[(train['Age'] > 40) & (train['Age'] <= 50), 'AgeGroup'] = 4

    dataset.loc[(train['Age'] > 10) & (train['Age'] <= 20), 'AgeGroup'] = 5

    dataset.loc[(train['Age'] > 20) & (train['Age'] <= 30), 'AgeGroup'] = 6

    dataset.loc[(train['Age'] > 70) & (train['Age'] <= 80), 'AgeGroup'] = 7

    dataset.loc[(train['Age'] > 60) & (train['Age'] <= 70), 'AgeGroup'] = 8

    

# Convert to integer type

train['AgeGroup'] = train['AgeGroup'].astype(int)

test['AgeGroup'] = test['AgeGroup'].astype(int)



# Remove the AgeBin column

train = train.drop(['AgeBin'], axis=1)# Investigate the survival probability after grouping the age into N categories (Here I use N=8)
### Sex ###



# Invastigate how sex is correlated to the survival probability

train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Invastigate how sex and age category is correlated to the survival probability

train[['Sex', 'AgeGroup', 'Survived']].groupby(['Sex','AgeGroup'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Create new feature "IsChild"

train['IsChild'] = 0

test['IsChild'] = 0

train.loc[train['Age'] <= 10, 'IsChild'] = 1

test.loc[test['Age'] <= 10, 'IsChild'] = 1



train.head()
### Embarked ###



# Count the number of missing value in "Embarked" column

count_nan_embarked_train = train["Embarked"].isnull().sum() # count_nan_embarked_train = 2

count_nan_embarked_test = test["Embarked"].isnull().sum()   # count_nan_embarked_test = 0



# Get the most common port from the data

freq_port_train = train.Embarked.dropna().mode()[0]

 

# Fill the missing values of the "Embarked" column with the most common port in the datasets

train['Embarked'] = train['Embarked'].fillna(freq_port_train)



train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).agg(['mean', 'std'])
### SibSp and Parch ###



# Create a new column "FamilySize" by adding the passenger with the number of his/her relatives aboard

full = [train, test]

for dataset in full:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1



train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean()
# Create new column "FamilySizeGroup" and assign "Alone", "Small" and "Big"

for dataset in full:

    dataset['FamilySizeGroup'] = 'Small'

    dataset.loc[dataset['FamilySize'] == 1, 'FamilySizeGroup'] = 'Alone'

    dataset.loc[dataset['FamilySize'] >= 5, 'FamilySizeGroup'] = 'Big'
### Cabin and Ticket ###



# Count the number of missing value in "Cabin" column

count_nan_cabin_train = train["Cabin"].isnull().sum() # count_nan_cabin_train = 687

count_nan_cabin_test = test["Cabin"].isnull().sum()   # count_nan_cabin_test = 327



# Drop the "Cabin" column as there are more than half of data has missing values 

# Drop the "ticket" column as this feature seems totally random

for dataset in full:

    dataset = dataset.drop(['Cabin','Ticket'], axis=1, inplace=True)
### Fare and Pclass ###



# Summarise the fare of different ticket classes

train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).agg(['mean', 'median', 'std', 'min', 'max'])
# Further investigate the survival chance for passengers with zero ticket fare

train.loc[(train.Fare == 0), ['Survived', 'Fare']]
# Get average, std, and number of NaN ages in training data

average_fare_train   = train["Fare"].mean()

std_fare_train       = train["Fare"].std()

count_nan_fare_train = train["Fare"].isnull().sum() # count_nan_fare_train = 0

count_nil_fare_train = (train["Fare"] == 0).sum()   # count_nil_fare_train = 15



# Get average, std, and number of NaN ages in test data

average_fare_test   = test["Fare"].mean()

std_fare_test       = test["Fare"].std()

count_nan_fare_test = test["Fare"].isnull().sum()  # count_nan_fare_test = 1

count_nil_fare_test = (test["Fare"] == 0).sum()    # count_nil_fare_test = 2



# Fill the missing fare values with the median of the respective ticket class

for dataset in full:

    dataset.loc[dataset.Fare.isnull(), 'Fare'] = dataset.groupby('Pclass').Fare.transform('median')

    #dataset.loc[(dataset.Fare == 0), 'Fare'] = dataset.groupby('Pclass').Fare.transform('median')    



train[['Pclass', 'Fare']].groupby(['Pclass'], as_index=False).agg(['mean', 'median', 'std', 'min', 'max'])
# Create new feature "FareGroup" based on the median fare of different "Pclass"

for dataset in full:

    dataset.loc[ dataset['Fare'] <= 8.05, 'FareGroup'] = 0

    dataset.loc[(dataset['Fare'] > 8.05) & (dataset['Fare'] <= 14.25), 'FareGroup'] = 1

    dataset.loc[(dataset['Fare'] > 14.25) & (dataset['Fare'] <= 60.2875), 'FareGroup']   = 2

    dataset.loc[ dataset['Fare'] > 60.2875, 'FareGroup'] = 3

    

    dataset['Fare'] = dataset['Fare'].astype(int)

    dataset['FareGroup'] = dataset['FareGroup'].astype(int)
title_mapping = {"Mr": 0, "Officer": 1, "Master": 2, "Miss": 3, "Royal": 4, "Mrs": 5}

sex_mapping = {"female": 0, "male": 1}

embarked_mapping = {"S": 0, "Q": 1, "C": 2}

family_mapping = {"Small": 0, "Alone": 1, "Big": 2}

for dataset in full:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Sex'] = dataset['Sex'].map(sex_mapping)

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping)

    dataset['FamilySizeGroup'] = dataset['FamilySizeGroup'].map(family_mapping)



    

train.head()
# Investigate the correlation of all the features with the "Survived"

train.corr()['Survived']



# Drop features in both train and test dataset

for dataset in full:

    dataset.drop('Name', axis=1, inplace=True)

    dataset.drop('FamilySize', axis=1, inplace=True)

    dataset.drop(['SibSp','Parch'], axis=1, inplace=True)     

    dataset.drop('Age', axis=1, inplace=True) 
# Prepare training and testing data

X_train = train.drop(['Survived', 'PassengerId'], axis=1)

Y_train = train['Survived']

X_test  = test.drop('PassengerId', axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape



# Standardise/Normalise the data

scaler = preprocessing.StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)



# Define variable for k-fold cross validation

k = 5



# Method to calculate training and k-fold accuracies

def calculate_accuracy(clf, X, Y, k):

    # Calculate training accuracy

    train_score = clf.score(X, Y)

    print("Training Accuracy: %0.2f" % train_score)



    # Calculate 5-fold cross validation accuracy

    scores = cross_val_score(clf, X, Y, cv=k)

    cv_score = scores.mean()

    print("5-Fold CV Accuracy: %0.2f (+/- %0.2f)" % (cv_score, scores.std() * 2))

    

    # Return the accuracies

    return train_score, cv_score
### Logistic Regression ###



# Build model and predict the test data

lr = LogisticRegression()

lr.fit(X_train, Y_train)

Y_pred_lr = lr.predict(X_test)



# Calculate accuracy

lr_train_score, lr_cv_score = calculate_accuracy(lr, X_train, Y_train, k)
### SVM (RBF Kernel)###



# Build model and predict the test data

svc = SVC(kernel='rbf')

svc.fit(X_train, Y_train)

Y_pred_svc = svc.predict(X_test)



# Calculate accuracy

svc_train_score, svc_cv_score = calculate_accuracy(svc, X_train, Y_train, k)
### KNN ###



# Build model and predict the test data

knn = KNeighborsClassifier(n_neighbors = 11)

knn.fit(X_train, Y_train)

Y_pred_knn = knn.predict(X_test)



# Calculate accuracy

knn_train_score, knn_cv_score = calculate_accuracy(knn, X_train, Y_train, k)
### Gaussian Naive Bayes ###



# Build model and predict the test data

nb = GaussianNB()

nb.fit(X_train, Y_train)

Y_pred_nb = nb.predict(X_test)



# Calculate accuracy

nb_train_score, nb_cv_score = calculate_accuracy(nb, X_train, Y_train, k)
### Multi Layer Perceptron ###



# Build model and predict the test data

mlp = MLPClassifier(solver='lbfgs', activation='logistic', alpha=1e-5, max_iter=1000,

                    hidden_layer_sizes=(5,3), learning_rate='adaptive',random_state=1)

mlp.fit(X_train, Y_train)

Y_pred_mlp = mlp.predict(X_test)



# Calculate accuracy

mlp_train_score, mlp_cv_score = calculate_accuracy(mlp, X_train, Y_train, k)
### Stochastic Gradient Descent ###



# Build model and predict the test data

sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred_sgd = sgd.predict(X_test)



# Calculate accuracy

sgd_train_score, sgd_cv_score = calculate_accuracy(sgd, X_train, Y_train, k)
### Decision Tree ###



# Build model and predict the test data

dt = DecisionTreeClassifier()

dt.fit(X_train, Y_train)

Y_pred_dt = dt.predict(X_test)



# Calculate accuracy

dt_train_score, dt_cv_score = calculate_accuracy(dt, X_train, Y_train, k)
### Random Forest ###



# Build model and predict the test data

rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train, Y_train)

Y_pred_rf = rf.predict(X_test)



# Calculate accuracy

rf_train_score, rf_cv_score = calculate_accuracy(rf, X_train, Y_train, k)
summary = pd.DataFrame({

    "Model": ["Support Vector Machines", "KNN", "Logistic Regression", 

              "Random Forest", "Naive Bayes", "Multi Layer Perceptron", 

              "Stochastic Gradient Decent", "Decision Tree"],

    "Training Score": [svc_train_score, knn_train_score, lr_train_score, 

              rf_train_score, nb_train_score, mlp_train_score, 

              sgd_train_score, dt_train_score],

    "CV Score": [svc_cv_score, knn_cv_score, lr_cv_score, 

              rf_cv_score, nb_cv_score, mlp_cv_score, 

              sgd_cv_score, dt_cv_score]})

summary.sort_values(by="CV Score", ascending=False)
# Prepare prediction result using svc result

prediction_result = pd.DataFrame({

        "PassengerId": test["PassengerId"],

        "Survived": Y_pred_svc

    })



# Save to csv file

#prediction_result.to_csv("".join((working_directory, 'output/submission_svc.csv')), index=False)