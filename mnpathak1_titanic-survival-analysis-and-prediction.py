# Import python libraries



import pandas as pd                       # for data processing, file I/O

from pandas import Series, DataFrame      

import numpy as np                        # linear algebra

import matplotlib.pyplot as plt           # visualization

import seaborn as sns                     # visualization

%matplotlib inline
# Import Machine Learning libraries:



# machine learning libraries

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
# Import dataset:



train_df=pd.read_csv('../input/train.csv')

test_df=pd.read_csv('../input/test.csv')
# Combine both train and test datset for data engineering:



combine_df = [train_df, test_df]
# Preview dataset:

train_df.info()

print('_'*40)

test_df.info()
train_df.describe()
test_df.describe()
train_df.describe(include=['O'])
test_df.describe(include=['O'])
train_df[["Pclass", "Survived"]].groupby(["Pclass"], as_index=False).mean().sort_values(by="Survived", ascending=False)
train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# use a facetgrid (FacetGrid is used to draw plots with multiple Axes 

# where each Axes shows the same relationship conditioned on different levels of some variable.

# It’s possible to condition on up to three variables by assigning variables 

# to the rows and columns of the grid and using different colors for the plot elements.)



g = sns.FacetGrid(train_df, col='Survived')

g.map(plt.hist, 'Age', bins=20)
generations = [10,20,40,60,80]

sns.lmplot('Age', 'Survived', hue='Sex', data=train_df, palette='winter', x_bins=generations, size=3.5, aspect=1.2)
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6 )

grid.map(plt.hist, 'Age')

grid.add_legend()
grid = sns.FacetGrid(train_df, col='Embarked', size=3, aspect=1.2)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived',size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=0.5, ci=None)

grid.add_legend()
print("Before", train_df.shape, test_df.shape, combine_df[0].shape, combine_df[1].shape)



train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine_df = [train_df, test_df]



print("After", train_df.shape, test_df.shape, combine_df[0].shape, combine_df[1].shape)
# iterate over both datasets in combine (train_df and test_df)

# and note that we also added the new field to both datasets



for dataset in combine_df:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

    

pd.crosstab(train_df['Title'], train_df['Sex'])
for dataset in combine_df:

    # these titles are interesting, but very few numbers of them exist, hence 'Rare'

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',\

       'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}



for dataset in combine_df:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

   

train_df.head()
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine_df = [train_df, test_df]



train_df.shape, test_df.shape
for dataset in combine_df:

    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    

train_df.head()
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=0.5, bins=20)

grid.add_legend()
# Preallocate an array to store an age guess for each gender, Pclass combination



guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine_df:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()

            #print(age_guess)

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            #print(guess_ages)

            #print('-'*10)

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),

                        'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)   



train_df.head(10) 
# Grouping Age column:



train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine_df:

    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[ (dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[ (dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[ (dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[  dataset['Age'] > 64, 'Age'] = 4
grid = sns.FacetGrid(train_df, col='Survived')

grid.map(plt.hist, 'Age', bins=20)
train_df = train_df.drop(['AgeBand'], axis=1)

combine_df = [train_df, test_df]

train_df.head()
for dataset in combine_df:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

    

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine_df:

    dataset['isAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'isAlone'] = 1 #set isAlone to False if family 



train_df[['isAlone', 'Survived']].groupby(['isAlone'], as_index=False).mean()   
# Eliminate the Parch, SibSp and FamilySize features and focus on isAlone for further analysis.



train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)



combine_df = [train_df, test_df]



train_df.head()
for dataset in combine_df:

    dataset['Age*Class'] = dataset.Age * dataset.Pclass

    

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)
freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
# use freq_port in the the fillna function

for dataset in combine_df:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

    

train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
for dataset in combine_df:

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2}).astype(int)

    

train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
train_df['Fare'].fillna(train_df['Fare'].dropna().median(), inplace=True)

train_df.head()
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
for dataset in combine_df:

    dataset.loc[ dataset['Fare'] <=7.91, 'Fare'] = 0

    dataset.loc[ (dataset['Fare'] > 7.91) & (dataset['Fare'] <=14.454), 'Fare' ] = 1

    dataset.loc[ (dataset['Fare'] > 14.454) & (dataset['Fare'] <=31), 'Fare' ] = 2

    dataset.loc[ (dataset['Fare'] > 31), 'Fare' ] = 3

    dataset['Fare'] = dataset['Fare'].astype(int)





train_df = train_df.drop(['FareBand'], axis=1)
train_df.head(10)
test_df.head(10)
colormap = plt.cm.viridis

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train_df.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
X_train = train_df.drop('Survived', axis=1) # independent variables only

Y_train = train_df['Survived']

X_test = test_df.drop('PassengerId', axis=1).copy()

X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df['Correlation'] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by='Correlation', ascending=False)
svc  = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Test k values 1 through 20

k_range = range(1, 21)



# Set an empty list

accuracy = []



# Repeat above process for all k values and append the result

for k in k_range:

    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, Y_train)

    Y_pred = knn.predict(X_test)

    

    accuracy.append(round(knn.score(X_train, Y_train) * 100, 2))
plt.plot(k_range, accuracy)

plt.xlabel('K value for for kNN')

plt.ylabel('Testing Accuracy')
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
feature_importance = random_forest.feature_importances_
# make importances relative to max importance and sort:

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0])
sorted_idx
pos
# Plot

plt.figure(figsize=(12, 7))

plt.subplot(1, 2, 2)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'kNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
results = Y_pred

results = pd.Series(results,name="Survived")

results=results.astype(int)

test_df = pd.read_csv('../input/test.csv')

passengerid = test_df['PassengerId'].astype(int)

submission5 = pd.DataFrame({"PassengerId": test_df['PassengerId'], "Survived": results})

submission5.to_csv("submission5_random_forest.csv",index=False)

from subprocess import check_output

print(check_output(["ls", "."]).decode("utf8"))