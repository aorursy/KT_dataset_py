# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

#matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



#set max columns and rows-display

pd.set_option("display.max_columns", 100)

pd.set_option("display.max_rows", 100)
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

df_list = [train_df, test_df] #list of df so actions quickly can be made on all df's
print(train_df.head()) #In reality I look at the DataFrame in Spyder.

print(train_df.columns.values)
print(train_df.info())

print("--------------------------")

print(test_df.info())
print(train_df.describe())
print(train_df.describe(include=['O']))
for df in df_list:

    A = df["Name"].str.split(",", expand = True)

    df["LastName"] = A[0].str.replace(' ', '')

    B = A[1].str.split(".", expand = True)    

    df["Title"] = B[0].str.replace(' ', '')

    df.drop(["PassengerId", "Ticket", "Name"], axis=1, inplace=True)
print(train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False)

                .mean().sort_values(by='Survived', ascending=False))



print(train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False)

                .mean().sort_values(by='Survived', ascending=False))



print(train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False)

               .mean().sort_values(by='Survived', ascending=False))
for df in df_list:

    df["FamilySize"] = df["SibSp"] + df["Parch"]

    df.drop(["SibSp", "Parch"], axis=1, inplace=True)

    

plt.figure()

sns.distplot(train_df.FamilySize)
for df in df_list:   

    df.loc[(df['FamilySize'] == 0), 'FamilySize'] = 0

    df.loc[(df['FamilySize'] == 1), 'FamilySize'] = 1

    df.loc[(df['FamilySize'] > 1) , 'FamilySize'] = 2
print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False)

                .mean().sort_values(by='Survived', ascending=False))
sns.FacetGrid(train_df, col='Survived').map(plt.hist, 'Age', bins=10)
print(pd.crosstab(train_df['Title'], train_df['Sex']))
plt.figure()

plt.figure(figsize=(20,10))

plt.title("Age vs Title")

sns.boxplot(x="Title",y="Age", data=train_df)
for df in df_list:

    df.loc[df.Sex == "female", "Title"] = df["Title"].apply(lambda title: "Miss" if (title == "Ms") or (title == "Mlle") or (title == "Mme") or (title == "Miss")

              else "Mrs")

    df.loc[df.Sex == "male", "Title"] = df["Title"].apply(lambda title: "Master" if (title == "Master")

              else "Mr")
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by='Survived', ascending=False))
grid = sns.FacetGrid(train_df, row='Pclass', col='Title', height=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4}

for df in df_list:

    df['Title'] = df['Title'].map(titles)

    df['Title'] = df['Title'].fillna(0)
guess_ages = np.zeros((4,3))

#print(guess_ages)





for df in df_list:

    for i in range(0, 4):

        for j in range(0, 3):

            guess_df = df[(df['Title'] == i+1) & (df['Pclass'] == j+1)]['Age'].dropna()



            age_mean = guess_df.mean()

            age_std = guess_df.std()

            age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5



    for i in range(0, 4):

        for j in range(0, 3):

            df.loc[ (df.Age.isnull()) & (df.Title == i+1) & (df.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    df['Age'] = df['Age'].astype(int)

    

#Finally - let's create age bands as we aim to create only numerical values for our model.

    

for df in df_list:   

    df.loc[ df['Age'] <= 12, 'Age'] = 0

    df.loc[(df['Age'] > 12) & (df['Age'] <= 18), 'Age'] = 1

    df.loc[(df['Age'] > 18) & (df['Age'] <= 35), 'Age'] = 2

    df.loc[(df['Age'] > 35) & (df['Age'] <= 64), 'Age'] = 3

    df.loc[ df['Age'] > 64, 'Age'] = 4  
print(train_df[['Age', 'Survived']].groupby(['Age'], as_index=False)

                .mean().sort_values(by='Survived', ascending=False))
##Convert Cabin to "Deck"

#for df in df_list:

#    df["Deck"] = df["Cabin"].str[0]

#    df.drop(["Cabin"], axis=1, inplace=True)

#This was a dead end -> So I decided to drop both Cabin and LastName.

for df in df_list:

    df.drop(["Cabin", "LastName"], axis=1, inplace=True)
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
plt.figure()

train_df[train_df["Embarked"].isnull()]["Fare"].value_counts().sort_index().plot.bar()

for df in df_list:

    df["Embarked"] = df["Embarked"].fillna(value="C")

    df['Embarked'] = df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)


guess_fare = np.zeros((6,3))

for df in df_list:

    for i in range(0, 6):

        for j in range(0, 3):

            guess_df = df[(df['Age'] == i) & (df['Pclass'] == j+1)]['Fare'].dropna()



            fare_mean = guess_df.mean()

            fare_std = guess_df.std()

            fare_guess = rnd.uniform(fare_mean - fare_std, fare_mean + fare_std)



            guess_fare[i,j] = float(fare_guess)



    for i in range(0, 6):

        for j in range(0, 3):

            df.loc[ (df.Fare.isnull()) & (df.Age == i) & (df.Pclass == j+1),\

                    'Fare'] = guess_fare[i,j]



    df['Fare'] = df['Fare'].astype(int)



plt.figure()

sns.kdeplot(train_df.Fare)



plt.figure()

sns.kdeplot(train_df.query('Fare < 200').Fare) 
for df in df_list:   

    df.loc[ df['Fare'] <= 10, 'Fare'] = 0

    df.loc[(df['Fare'] > 10) & (df['Fare'] <= 50), 'Fare'] = 1

    df.loc[(df['Fare'] > 50) & (df['Fare'] <= 100), 'Fare'] = 2

    df.loc[(df['Fare'] > 100) & (df['Fare'] <= 200), 'Fare'] = 3

    df.loc[ df['Fare'] > 200, 'Fare'] = 4 
print(train_df[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Survived', ascending=False))

X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)



coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



print(coeff_df.sort_values(by='Correlation', ascending=False))



# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)



# Pattern Recognition

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)



# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)



# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)



# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)



# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)



# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)



# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

print(models.sort_values(by='Score', ascending=False))