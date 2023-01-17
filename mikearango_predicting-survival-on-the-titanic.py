# Data Analysis and Wrangling

import pandas as pd

import numpy as np

import random as rnd



# Visualization

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



# Machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV

from sklearn.feature_selection import RFECV

from sklearn.cross_validation import train_test_split , StratifiedKFold

import xgboost as xgb





# Configure Visualizations

%matplotlib inline

mpl.style.use('ggplot')

sns.set_style('white')

pylab.rcParams['figure.figsize'] = 8 , 6
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]
print(train_df.columns.values)

train_df.head()

# train_df.tail()
train_df.info()

print('_'*40)

test_df.info()
print('In the train dataset:')

print('Age has ' + str(891 - 714) + ' missing values') 

print('Cabin has ' + str(891 - 204) + ' missing values') 

print('Embarked has ' + str(891 - 889) + ' missing values') 

print('_'*40)

print('In the test dataset:')

print('Age has ' + str(418 - 332) + ' missing values') 

print('Fare has ' + str(418 - 417) + ' missing values')

print('Cabin has ' + str(418 - 91) + ' missing values')
def plotHist(df, x, col, row = None, bins = 20): 

    grid = sns.FacetGrid(df, col = col, row = row)

    grid.map(plt.hist, x, bins = bins)

    

def plotDistribution(df, var, target, **kwargs):

    rowVar = kwargs.get('rowVar', None)

    colVar = kwargs.get('colVar' , None)

    grid = sns.FacetGrid(df, hue = target, aspect = 4, row = rowVar, col = colVar)

    grid.map(sns.kdeplot, var, shade = True)

    grid.set(xlim = (0, df[var].max()))

    grid.add_legend()



def plotCategorical(df, cat, target, **kwargs):

    rowVar = kwargs.get('rowVar', None)

    colVar = kwargs.get('colVar', None)

    grid = sns.FacetGrid(df, row = rowVar, col = colVar)

    grid.map(sns.barplot, cat, target)

    grid.add_legend()



def plotCorrelation(df):

    corr = df.corr()

    heat, ax = plt.subplots(figsize = (12, 10))

    cmap = sns.diverging_palette(220, 10, as_cmap = True)

    heat = sns.heatmap(

        corr, 

        cmap = cmap,

        square = True, 

        cbar_kws = {'shrink': .9}, 

        ax = ax, 

        annot = True, 

        annot_kws = {'fontsize': 12})



def describeMore(df):

    var = [] ; l = [] ; t = []

    for x in df:

        var.append(x)

        l.append(len(pd.value_counts(df[x])))

        t.append(df[x].dtypes)

    levels = pd.DataFrame({'Variable': var, 'Levels': l, 'Datatype': t})

    levels.sort_values(by = 'Levels', inplace = True)

    return levels



def analyzeByPivot(df, grouping_var, sorting_var):

    place = df[[grouping_var, sorting_var]].groupby([grouping_var], 

               as_index = False).mean().sort_values(by = sorting_var, ascending = False)

    return place
describeMore(train_df)
train_df.describe(percentiles = [.01, .05, .10, .25, .50, .75, .90, .95, .99])
train_df.describe(include=['O'])
plotCorrelation(train_df)
analyzeByPivot(train_df, 'Pclass', 'Survived')
analyzeByPivot(train_df, 'Sex', 'Survived')
analyzeByPivot(train_df, 'SibSp', 'Survived')
analyzeByPivot(train_df, 'Parch', 'Survived')
plotCategorical(train_df, 'Pclass', 'Survived')
plotHist(train_df, 'Age', 'Survived')
plotCategorical(train_df, 'SibSp', 'Survived')
plotCategorical(train_df, 'Parch', 'Survived')
plotHist(train_df, 'Age', 'Survived', 'Pclass', bins = 20)
plotHist(train_df, 'Age', 'Survived', row = 'Sex')

plotDistribution(train_df, var = 'Age', target = 'Survived', rowVar = 'Sex')
sns.pointplot(x = "Pclass", y = "Survived", hue = "Sex", data = train_df,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"])
grid = sns.FacetGrid(train_df, col='Embarked')

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
plotCategorical(train_df, 'Sex', 'Fare', rowVar = 'Embarked', colVar = 'Survived')



# grid = sns.FacetGrid(train_df, row = 'Embarked', col = 'Survived', size = 2.2, aspect = 1.6)

# grid.map(sns.barplot, 'Sex', 'Fare', ci = None)

# grid.add_legend()
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)

test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)

combine = [train_df, test_df]



print("After dropping Ticket and Cabin -- ", 

'Train:', train_df.shape, ', Test:', test_df.shape)
for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(train_df['Title'], train_df['Sex'])
analyzeByPivot(train_df, 'Title', 'Survived')
for dataset in combine:

    # Standardize officer titles 

    dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Major', 'Dr'], 'Officer')

    # Standardize royal titles

    dataset['Title'] = dataset['Title'].replace(['Jonkheer', 'Don', 'Sir', 'Countess', 'Dona', 'Lady'],

                                                'Royalty')

    # Mlle stands for Mademoiselle which is French for Miss

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    # Standardize Ms to Miss

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    # Mme stands for Madame which is French for Mrs

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
analyzeByPivot(train_df, 'Title', 'Survived')
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)

test_df = test_df.drop(['Name'], axis=1)

combine = [train_df, test_df]

train_df.shape, test_df.shape
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royalty": 5, "Officer": 6, "Rev": 7}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    # We only checked train_df for titles, so we don't know if there are other non-mapped titles in 

    # test_df. For this reason we fill NA's with 0 and reserve that as an "Other" level.

    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

train_df.head()
guess_ages = np.zeros((2,3))

guess_ages
# For each dataset

for dataset in combine:

    # For each sex

    for i in range(0, 2):

        # For each Pclass

        for j in range(0, 3):

            # Populate guess_age dataframe with each combo of Sex and Pclass for each iteration

            # and drop all NA values

            guess_df = dataset[(dataset['Sex'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # Set age_guess equal to the median age of each Sex and Pclass combo

            age_guess = guess_df.median()

            # Populate guess_ages array with each age_guess 

            guess_ages[i,j] = age_guess          

    # For each sex 

    for i in range(0, 2):

        # For each Pclass

        for j in range(0, 3):

            # For each combo of Sex and Pclass, find the rows where Age is null and impute with 

            # the corresponding age guess from the guess_ages array

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]

    # Convert age from float to int 

    dataset['Age'] = dataset['Age'].astype(int)



train_df.head()
plotDistribution(train_df, var = 'Age', target = None)

train_df.Age.describe()
# Cuts Age into 5 bins

train_df['AgeBin'] = pd.cut(train_df['Age'], 5)

analyzeByPivot(train_df, 'AgeBin', 'Survived')
combine = [train_df, test_df]



for dataset in combine: 

    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

train_df.head()
train_df = train_df.drop(['AgeBin'], axis=1)

combine = [train_df, test_df]

train_df.head()
for dataset in combine:

    dataset['GroupSize'] = dataset['SibSp'] + dataset['Parch'] + 1

analyzeByPivot(train_df, 'GroupSize', 'Survived')
for dataset in combine: 

    # GroupSize = 1 --> Alone

    dataset.loc[dataset['GroupSize'] == 1, 'GroupSize'] = 0

    # GroupSize = 2, 3, or 4 --> Small   

    dataset.loc[(dataset['GroupSize'] > 1) & (dataset['GroupSize'] <= 4), 'GroupSize'] = 1

    # GroupSize = 5, 6, or 7 --> Medium   

    dataset.loc[(dataset['GroupSize'] > 4) & (dataset['GroupSize'] <= 7), 'GroupSize'] = 2

    # GroupSize = 8+ --> Large  

    dataset.loc[dataset['GroupSize'] > 7, 'GroupSize'] = 3

analyzeByPivot(train_df, 'GroupSize', 'Survived')
train_df.head()
train_df = train_df.drop(['Parch', 'SibSp'], axis=1)

test_df = test_df.drop(['Parch', 'SibSp'], axis=1)

combine = [train_df, test_df]



train_df.head()
# Drop NA values and report to mode of Embark

freq_port = train_df.Embarked.dropna().mode()[0]

freq_port
for dataset in combine:

    # Impute missing values with "S"

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

analyzeByPivot(train_df, "Embarked", "Survived")
embarked_mapping = {'S': 0, 'C': 1, 'Q': 2}

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].map(embarked_mapping).astype(int)



train_df.head()
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

test_df.head()
train_df.Fare.describe(percentiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9])
train_df['FareBand'] = pd.qcut(train_df['Fare'], 5)

analyzeByPivot(train_df, 'FareBand', 'Survived')
for dataset in combine:

    dataset.loc[ dataset['Fare'] <= 7.854, 'Fare'] = 0

    dataset.loc[(dataset['Fare'] > 7.854) & (dataset['Fare'] <= 10.5), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 10.5) & (dataset['Fare'] <= 21.679), 'Fare'] = 2

    dataset.loc[(dataset['Fare'] > 21.679) & (dataset['Fare'] <= 39.688), 'Fare'] = 3

    dataset.loc[dataset['Fare'] > 39.688, 'Fare'] = 4

    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)

combine = [train_df, test_df]
train_df.head(10)
test_df.head(10)
plotCorrelation(train_df)
X_train = train_df.drop("Survived", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()



X_train.shape, Y_train.shape, X_test.shape
logreg = LogisticRegression() # Define Logistic Regression model

logreg.fit(X_train, Y_train) # Run the model using the training data

Y_pred = logreg.predict(X_test) # Use the model to predict Y values given the test data

acc_log = round(logreg.score(X_train, Y_train) * 100, 2) # Accuracy 

print("Accuracy:", acc_log)
rfecv = RFECV(estimator = logreg, step = 1, cv = StratifiedKFold(Y_train, 2 ), scoring = 'accuracy')

rfecv.fit(X_train,Y_train)

print("Accuracy:", round((rfecv.score(X_train, Y_train) * 100),2))

print("Optimal number of variables: %d" % rfecv.n_features_)



# Plot number of variables VS. cross-validation scores

plt.figure()

plt.xlabel("Number of Variables Used")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1 , len(rfecv.grid_scores_) + 1), rfecv.grid_scores_ * 100)

plt.show()
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Variable']

coeff_df["Correlation Coeff"] = pd.Series(logreg.coef_[0])

coeff_df.sort_values(by = 'Correlation Coeff', ascending=False)
svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

print("Accuracy:", acc_svc)
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

print("Accuracy:", acc_linear_svc)
rfecv = RFECV(estimator = linear_svc, step = 1, cv = StratifiedKFold(Y_train, 2 ), scoring = 'accuracy')

rfecv.fit(X_train,Y_train)

print("Accuracy:", round((rfecv.score(X_train, Y_train) * 100),2))

print("Optimal number of variables: %d" % rfecv.n_features_)



# Plot number of variables VS. cross-validation scores

plt.figure()

plt.xlabel("Number of Variables Used")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1 , len(rfecv.grid_scores_) + 1), rfecv.grid_scores_ * 100)

plt.show()
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

print("Accuracy: ", acc_knn)
gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

print("Accuracy: ", acc_gaussian)
perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

print("Accuracy: ", acc_perceptron)
rfecv = RFECV(estimator = perceptron, step = 1, cv = StratifiedKFold(Y_train, 2 ), scoring = 'accuracy')

rfecv.fit(X_train,Y_train)

print("Accuracy:", round((rfecv.score(X_train, Y_train) * 100),2))

print("Optimal number of variables: %d" % rfecv.n_features_)



# Plot number of variables VS. cross-validation scores

plt.figure()

plt.xlabel("Number of Variables Used")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1 , len(rfecv.grid_scores_) + 1), rfecv.grid_scores_ * 100)

plt.show()
sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

print("Accuracy: ", acc_sgd)
rfecv = RFECV(estimator = sgd, step = 1, cv = StratifiedKFold(Y_train, 2 ), scoring = 'accuracy')

rfecv.fit(X_train,Y_train)

print("Accuracy:", round((rfecv.score(X_train, Y_train) * 100),2))

print("Optimal number of variables: %d" % rfecv.n_features_)



# Plot number of variables VS. cross-validation scores

plt.figure()

plt.xlabel("Number of Variables Used")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1 , len(rfecv.grid_scores_) + 1), rfecv.grid_scores_ * 100)

plt.show()
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

print("Accuracy: ", acc_decision_tree)
rfecv = RFECV(estimator = decision_tree, step = 1, cv = StratifiedKFold(Y_train, 2 ), scoring = 'accuracy')

rfecv.fit(X_train,Y_train)

print("Accuracy:", round((rfecv.score(X_train, Y_train) * 100),2))

print("Optimal number of variables: %d" % rfecv.n_features_)



# Plot number of variables VS. cross-validation scores

plt.figure()

plt.xlabel("Number of Variables Used")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1 , len(rfecv.grid_scores_) + 1), rfecv.grid_scores_ * 100)

plt.show()
def plot_model_var_imp(model, X, y):

    imp = pd.DataFrame( 

        model.feature_importances_  , 

        columns = [ 'Importance' ] , 

        index = X.columns 

    )

    imp = imp.sort_values([ 'Importance'], ascending = True)

    imp[:10].plot(kind = 'barh')



plot_model_var_imp(decision_tree, X_train, Y_train)
random_forest = RandomForestClassifier(n_estimators = 100)

random_forest.fit(X_train, Y_train)

Y_pred_rf = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print("Accuracy: ", acc_random_forest)
rfecv = RFECV(estimator = random_forest, step = 1, cv = StratifiedKFold(Y_train, 2 ), scoring = 'accuracy')

rfecv.fit(X_train,Y_train)

print("Accuracy:", round((rfecv.score(X_train, Y_train) * 100),2))

print("Optimal number of variables: %d" % rfecv.n_features_)



# Plot number of variables VS. cross-validation scores

plt.figure()

plt.xlabel("Number of Variables Used")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1 , len(rfecv.grid_scores_) + 1), rfecv.grid_scores_ * 100)

plt.show()
grad_boost = GradientBoostingClassifier()

grad_boost.fit(X_train, Y_train)

Y_pred = grad_boost.predict(X_test)

grad_boost.score(X_train, Y_train)

acc_grad_boost = round(grad_boost.score(X_train, Y_train) * 100, 2)

print("Accuracy: ", acc_grad_boost)
rfecv = RFECV(estimator = linear_svc, step = 1, cv = StratifiedKFold(Y_train, 2 ), scoring = 'accuracy')

rfecv.fit(X_train,Y_train)

print("Accuracy:", round((rfecv.score(X_train, Y_train) * 100),2))

print("Optimal number of variables: %d" % rfecv.n_features_)



# Plot number of variables VS. cross-validation scores

plt.figure()

plt.xlabel("Number of Variables Used")

plt.ylabel("Cross validation score (nb of correct classifications)")

plt.plot(range(1 , len(rfecv.grid_scores_) + 1), rfecv.grid_scores_ * 100)

plt.show()
xgboost = xgb.XGBClassifier(max_depth = 3, n_estimators = 300, learning_rate = 0.05)

xgboost.fit(X_train, Y_train)

Y_pred = xgboost.predict(X_test)

xgboost.score(X_train, Y_train)

acc_xgboost = round(xgboost.score(X_train, Y_train) * 100, 2)

print("Accuracy: ", acc_xgboost)
models = pd.DataFrame({

    'Model': ['Logistic Regression', 'Support Vector Classifier', 'Linear SVC', 'k-NN', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 

              'Decision Tree', "Gradient Boosting Classifier", "XGBoost"],

    'Score': [acc_log, acc_svc, acc_linear_svc, acc_knn, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_decision_tree, acc_grad_boost, acc_xgboost]})

models.sort_values(by = 'Score', ascending = False)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred_rf})

submission.to_csv('submission.csv', index = False)

submission.head()