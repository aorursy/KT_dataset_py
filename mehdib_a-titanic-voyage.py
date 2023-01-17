# Imports



# pandas

import pandas as pd



# numpy, matplotlib

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d, Axes3D

%matplotlib inline



# machine learning

import sklearn

from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB
# Helper functions



# Estimation function for any classifier

def estimate(clf, data, target, iters=5):

    scores = cross_val_score(clf, data, target, cv=iters)

    return (scores.mean(), scores.std()*2)



# Predict and Y pred from any classifier

def get_pred(clf, data, target, test):

    return clf.fit(data, target).predict(test)
# Import data



# get titanic & test csv files as a DataFrame

titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")

full_df = pd.concat([titanic_df, test_df], keys=['training', 'test'])



# preview the data

#full_df.head(5)
# Info



#full_df.info()
# Basic description



#full_df.describe()
# Title



#full_df['Name'].head(10)

full_df['Title'] = full_df['Name'].str.replace('(.*, )|(\\..*)', '')
counts = full_df['Title'].value_counts() # Mr, Miss, Mrs, Master are most common

all_titles = counts.index.values

keep_titles = ['Mr', 'Miss', 'Mrs', 'Master', 'Rare']

del_titles = np.setdiff1d(all_titles, keep_titles, assume_unique=True)

print(all_titles, keep_titles, del_titles)



full_df.replace('Mlle', 'Miss', inplace=True)

full_df.replace('Ms', 'Miss', inplace=True)

full_df.replace('Mme', 'Mrs', inplace=True)

full_df.replace(del_titles, 'Rare', inplace=True)



title_dummies = pd.get_dummies(full_df.Title)



full_df = full_df.drop(['Title'], axis=1)

full_df = full_df.join(title_dummies)
# Fare



#full_df[full_df.Fare.isnull()] # Result : 1 NaN Fare

full_df.Fare = full_df.Fare.fillna(full_df.Fare.median())

#full_df[full_df.Fare.isnull()] # Result : No more NaN

full_df.Fare = full_df.Fare.astype(int)

#hist = full_df.Fare.hist(bins=10)
# Pclass



# We create dummy variables for Pclass

# A=[X OR Y OR Z] => 3 cols (X,Y,Z) are created, if A=X: X=1, Y=0, Z=0



pclass_dummies = pd.get_dummies(full_df.Pclass)

pclass_dummies.columns = ['Class1', 'Class2', 'Class3']

pclass_dummies = pclass_dummies.drop(['Class3'], axis=1) # To drop or not to drop ?



full_df = full_df.drop(['Pclass'], axis=1)

full_df = full_df.join(pclass_dummies)
# Age



# Creating random ages to fill NaN ages

average_age_titanic = full_df.Age.mean()

std_age_titanic = full_df.Age.std()

count_nan_age_titanic = full_df.Age.isnull().sum()

rand_age = np.random.randint(low=average_age_titanic-std_age_titanic, high=average_age_titanic+std_age_titanic, size=count_nan_age_titanic)



# Replacing NaN with random ages

if full_df[full_df.Age.isnull()].values.any():

    full_df['Age'][full_df.Age.isnull()] = rand_age



# Converting to int for better classifying

full_df["Age"] = full_df["Age"].astype(int)



# Drawing histogram with random ages

#hist = full_df.Age.hist(bins=20, range=(0,80))
# Person

# As we see, children(age < ~16) on aboard seem to have a high chances for Survival.

# So, we can classify passengers as males, females, and child

def get_person(passenger):

    age,sex = passenger

    return 'child' if age < 16 else sex

full_df['Person'] = full_df[['Age','Sex']].apply(get_person,axis=1)



# Converting Person to int for better classification

person_dummies = pd.get_dummies(full_df['Person'])

person_dummies.columns = ['Child', 'Female', 'Male']

person_dummies = person_dummies.drop(['Male'], axis=1)



full_df = full_df.join(person_dummies)

full_df = full_df.drop(['Person'], axis=1)
# Family



# Instead of having two columns Parch & SibSp, 

# we can have only one column represent if the passenger had any family member aboard or not,

# Meaning, if having any family member(whether parent, brother, ...etc) will increase chances of Survival or not.

full_df['Family'] =  full_df["Parch"] + full_df["SibSp"]



# Adding other attribute

full_df['Family'].loc[full_df['Family'] > 0] = 1

full_df['Family'].loc[full_df['Family'] == 0] = 0
# Preparing final dataset

full_df_f = full_df.drop(['SibSp', 'Parch', 'PassengerId', 'Ticket', 'Cabin', 

                          'Name', 'Sex', 'Embarked'], axis=1, errors='ignore')



# define training and testing sets

X_train = full_df_f.ix['training'].drop("Survived",axis=1)

X_train = X_train.dropna().astype(int) # Dropping NaN survival rows

Y_train = full_df_f.ix['training']["Survived"]

Y_train = Y_train.dropna().astype(int)

X_test  = full_df_f.ix['test'].drop("Survived", axis=1)
# Logistic Regression



logreg = LogisticRegression()
# Support Vector Machines



svc = SVC()
# Random Forest Classifier



random_forest = RandomForestClassifier(n_estimators=10)  
# K-Neighbors Classifier



knn = KNeighborsClassifier(n_neighbors = 10)
# PCA decomposition



from sklearn.decomposition import PCA

pca = PCA(n_components=3).fit(X_train, Y_train).transform(X_train)

pca = np.transpose(pca)



pca2 = PCA(n_components=2).fit(X_train, Y_train).transform(X_train)

pca2 = np.transpose(pca2)



fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(pca[0], pca[1], pca[2], c='r', marker='o')

plt.show()



fig2 = plt.figure()

ax = fig2.add_subplot(111)

ax.scatter(pca2[0], pca2[1], c='r', marker='o')

plt.show()
import seaborn as sns



corr = full_df.drop(['PassengerId'], axis=1).corr()

cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(8, 5))



# Setting heatmap vmax, vmin (excluding -1,1 correlations)

vmin = corr[corr > -1.0].min().min()

vmax = corr[corr < 1.0].max().max()



# Draw the heatmap using seaborn

hmp = sns.heatmap(corr, vmin=vmin, vmax=vmax, square=True)
gaussian = GaussianNB()
random_forest = RandomForestClassifier(n_estimators=53)
Y_pred = get_pred(random_forest, X_train, Y_train, X_test)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

submission.to_csv('titanic.csv', index=False)
submission.head()