#import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline





# machine learning

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# Import Movies Dataset

dfMovies = pd.read_csv("../input/movies.dat",sep="::",names=["MovieID","Title","Genres"],engine='python')

dfMovies.head()
# Import Ratings Dataset

dfRatings = pd.read_csv("../input/ratings.dat",sep="::",names=["UserID","MovieID","Rating","Timestamp"],engine='python')

dfRatings.head()
# Import Ratings Dataset

dfUsers = pd.read_csv("../input/users.dat",sep="::",names=["UserID","Gender","Age","Occupation","Zip-code"],engine='python')

dfUsers.head()
dfMovies.shape
dfUsers.shape
dfRatings.shape
dfMovieRatings = dfMovies.merge(dfRatings,on='MovieID',how='inner')

dfMovieRatings.head()
# to check whether merging does not changes any dataset

dfMovieRatings.shape
dfMaster = dfMovieRatings.merge(dfUsers,on="UserID",how='inner')

dfMaster.head()
dfMaster.to_csv("Master.csv")
# Users with Different Age Groups

dfMaster['Age'].value_counts()
# Plot for users with different age groups

dfMaster['Age'].value_counts().plot(kind='bar')

plt.xlabel("Age")

plt.title("User Age Distribution")

plt.ylabel('Users Count')

plt.show()
# Toy Story

toystoryRating = dfMaster[dfMaster['Title'].str.contains('Toy Story') == True]

toystoryRating
toystoryRating.groupby(["Title","Rating"]).size()
toystoryRating.groupby(["Title","Rating"]).size().unstack().plot(kind='barh',stacked=False,legend=True)

plt.show()
dfTop25 = dfMaster.groupby('Title').size().sort_values(ascending=False)[:25]

dfTop25
dfTop25.plot(kind='barh',alpha=0.6,figsize=(7,7))

plt.xlabel("Viewership Ratings Count")

plt.ylabel("Movies (Top 25)")

plt.title("Top 25 movies by viewership rating")

plt.show()

userId = 2696

userRatingById = dfMaster[dfMaster["UserID"] == userId]

userRatingById
#dfGenres = dfMaster[]

dfGenres = dfMaster['Genres'].str.split("|")
dfGenres


listGenres = set()

for genre in dfGenres:

    listGenres = listGenres.union(set(genre))
# All Unique genres

listGenres
ratingsOneHot = dfMaster['Genres'].str.get_dummies("|")
ratingsOneHot.head()
dfMaster = pd.concat([dfMaster,ratingsOneHot],axis=1)
dfMaster.head()
dfMaster.columns
dfMaster.to_csv("Final_Master.csv")
dfMaster[["title","Year"]] = dfMaster.Title.str.extract("(.)\s\((.\d+)",expand=True)
dfMaster = dfMaster.drop(columns=["title"])

dfMaster.head()
dfMaster.info()
dfMaster['Year'] = dfMaster.Year.astype(int)
dfMaster['Movie_Age'] = 2000 - dfMaster.Year

dfMaster.head()
dfMaster['Gender'] = dfMaster.Gender.str.replace('F','1')
dfMaster['Gender'] = dfMaster.Gender.str.replace('M','0')
dfMaster['Gender'] = dfMaster.Gender.astype(int)
dfMaster.head()
dfGenderAffecting = dfMaster.groupby('Gender').size().sort_values(ascending=False)[:25]

dfTest
dfMaster.groupby(["Gender","Rating"]).size().unstack().plot(kind='bar',stacked=False,legend=True)

plt.show()
dfMaster.groupby(["Age","Rating"]).size().unstack().plot(kind='bar',stacked=False,legend=True)

plt.show()
dfMaster.groupby(["Occupation","Rating"]).size().unstack().plot(kind='bar',stacked=False,legend=True)

plt.show()
dfMaster.groupby(["Year","Rating"]).size().unstack().plot(kind='bar',stacked=False,legend=True)

plt.show()
dfMaster.groupby(["Movie_Age","Rating"]).size().unstack().plot(kind='bar',stacked=False,legend=True)

plt.show()
#First 500 extracted records

first_500 = dfMaster[:1000]
first_500
#Use the following features:movie id,age,occupation

features = first_500[['MovieID','Age','Occupation']].values
#Use rating as label

labels = first_500[['Rating']].values
features
labels
#Create train and test data set

train, test, train_labels, test_labels = train_test_split(features,labels,test_size=0.33,random_state=42)
train
test
train_labels
test_labels
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(train, train_labels)

Y_pred = logreg.predict(test)

acc_log = round(logreg.score(train, train_labels) * 100, 2)

acc_log
# Support Vector Machines



svc = SVC()

svc.fit(train, train_labels)

Y_pred = svc.predict(test)

acc_svc = round(svc.score(train, train_labels) * 100, 2)

acc_svc
# K Nearest Neighbors Classifier



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(train, train_labels)

Y_pred = knn.predict(test)

acc_knn = round(knn.score(train, train_labels) * 100, 2)

acc_knn
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(train, train_labels)

Y_pred = gaussian.predict(test)

acc_gaussian = round(gaussian.score(train, train_labels) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(train, train_labels)

Y_pred = perceptron.predict(test)

acc_perceptron = round(perceptron.score(train, train_labels) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(train, train_labels)

Y_pred = linear_svc.predict(test)

acc_linear_svc = round(linear_svc.score(train, train_labels) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(train, train_labels)

Y_pred = sgd.predict(test)

acc_sgd = round(sgd.score(train, train_labels) * 100, 2)

acc_sgd
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(train, train_labels)

Y_pred = decision_tree.predict(test)

acc_decision_tree = round(decision_tree.score(train, train_labels) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(train, train_labels)

Y_pred = random_forest.predict(test)

random_forest.score(train, train_labels)

acc_random_forest = round(random_forest.score(train, train_labels) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)