import pandas as pd

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder

import numpy as np



# Load the data

data_train = pd.read_csv('../input/train.csv')

data_test = pd.read_csv('../input/test.csv')
data_train.info()
data_test.info()
y = data_train.Survived

testID = data_test.PassengerId
combine = [data_train, data_test]
# Visualisation

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns



def plot_correlation_map( df ):

    corr = data_train.corr()

    _ , ax = plt.subplots( figsize =( 12 , 10 ) )

    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )

    _ = sns.heatmap(

        corr, 

        cmap = cmap,

        square=True, 

        cbar_kws={ 'shrink' : .9 }, 

        ax=ax, 

        annot = True, 

        annot_kws = { 'fontsize' : 12 }

    )

plot_correlation_map( data_train )
g = sns.pairplot(data_train[[u'Survived', u'Pclass', u'Sex', u'Age', u'Parch', u'Fare', u'Embarked']], hue='Survived', palette = 'seismic',size=2.0,diag_kind = 'kde',diag_kws=dict(shade=True),plot_kws=dict(s=10) )

g.set(xticklabels=[])
for dataset in combine:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
pd.crosstab(data_train['Title'],data_train['Survived'])
pd.crosstab(data_test['Title'],data_test['Sex'])
for dataset in combine:

    dataset['Name_length'] = dataset['Name'].apply(len) - dataset['Title'].apply(len)

    

data_train['Name_length_Band'] = pd.qcut(data_train['Name_length'], 6)

data_train[['Name_length_Band','Survived']].groupby(['Name_length_Band'], as_index=False).mean().sort_values(by='Name_length_Band', ascending=True)
def simplify_name(df):

    bins = (0, 16.0, 19.0, 23.0, 25.0, 30.0, 79.0)

    group_names = ['0', '1', '2', '3', '4', '5']

    categories = pd.cut(df.Name_length, bins, labels=group_names)

    df.Name_length = categories

    return df



data_train = simplify_name(data_train)

data_train = data_train.drop(['Name_length_Band'], axis=1)

data_test = simplify_name(data_test)

combine = [data_train, data_test]

data_train.head()
Title_Dictionary = {

                    "Capt":       "Officer",

                    "Col":        "Officer",

                    "Major":      "Officer",

                    "Countess":   "Royalty",

                    "Jonkheer":   "Royalty",

                    "Don":        "Royalty",

                    "Sir" :       "Royalty",

                    "Dr":         "Officer",

                    "Rev":        "Officer",

                    "Countess":   "Royalty",

                    "Dona":       "Royalty",

                    "Mme":        "Mrs",

                    "Mlle":       "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mr",

                    "Mrs" :       "Mrs",

                    "Miss" :      "Miss",

                    "Master" :    "Master",

                    "Lady" :      "Royalty"

                    }

for dataset in combine:

    dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

    dataset['Title'] = dataset['Title'].map(Title_Dictionary)
data_train.head()
for dataset in combine:

    dataset['IsAlone'] = 'N'

    dataset.loc[dataset['SibSp'] + dataset['Parch'] + 1 == 1, 'IsAlone'] = 'Y'
pd.crosstab(data_train['IsAlone'],data_train['Survived'])
data_train[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='IsAlone', ascending=True)
data_train['Age'].fillna(data_train['Age'].dropna().median(), inplace=True)

data_train['AgeBand'] = pd.cut(data_train['Age'], 6)

data_train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
def simplify_ages(df):

    df.Age.fillna(df.Age.dropna().median(), inplace=True)

    bins = (0, 13.683, 26.947, 40.21, 53.473, 66.737, 80.0)

    group_names = ['0', '1', '2', '3', '4', '5']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df



data_train = simplify_ages(data_train)

data_train = data_train.drop(['AgeBand'], axis=1)

data_test = simplify_ages(data_test)

combine = [data_train, data_test]

data_train.head()
#data_train['Fare'].fillna(data_train['Fare'].dropna().median(), inplace=True)

data_train['FareBand'] = pd.qcut(data_train['Fare'], 5, precision=3)

data_train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
def simplify_fare(df):

    df.Fare.fillna(df.Fare.dropna().median(), inplace=True)

    bins = (-0.001, 7.854, 10.5, 21.679, 39.688, 512.329)

    group_names = ['0', '1', '2', '3', '4']

    categories = pd.cut(df.Fare, bins, labels=group_names, precision=3)

    df.Fare = categories

    return df



data_train = simplify_fare(data_train)

data_train = data_train.drop(['FareBand'], axis=1)

data_test = simplify_fare(data_test)

combine = [data_train, data_test]

data_train.head()
pd.crosstab(data_train['Embarked'],data_train['Survived'])
freq_port = data_train.Embarked.dropna().mode()[0]

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
data_train['Has_Cabin'] = data_train["Cabin"].apply(lambda x: 'N' if type(x) == float else 'Y')

data_test['Has_Cabin'] = data_test["Cabin"].apply(lambda x: 'N' if type(x) == float else 'Y')

data_train[['Has_Cabin', 'Survived']].groupby(['Has_Cabin'], as_index=False).mean().sort_values(by='Has_Cabin', ascending=True)
pd.crosstab(data_train['Has_Cabin'],data_train['Survived'])
data_train['Pclass'] = data_train['Pclass'].astype(str)

data_test['Pclass'] = data_test['Pclass'].astype(str)
data_train = data_train.drop(['Survived'], axis=1)

full = data_train.append( data_test , ignore_index = True )
full.head()
full = full.drop(['Name'], axis=1)

full = full.drop(['SibSp'], axis=1)

full = full.drop(['Parch'], axis=1)

full = full.drop(['Ticket'], axis=1)

full = full.drop(['Cabin'], axis=1)

full = full.drop(['PassengerId'], axis=1)
full.info()
full = pd.get_dummies(full)
full.head()
# Differentiate numerical features (minus the target) and categorical features

categorical_features = full.select_dtypes(include = ["object"]).columns

numerical_features = full.select_dtypes(exclude = ["object"]).columns

print("Numerical features : " + str(len(numerical_features)))

print("Categorical features : " + str(len(categorical_features)))
# Create all datasets that are necessary to train, validate and test models

from sklearn.model_selection import train_test_split



train_valid_X = full[ 0:891 ]

train_valid_y = y

X_test = full[ 891: ]



X_train, X_val, Y_train, Y_val = train_test_split( train_valid_X , train_valid_y , train_size = .90)
# PCA features

from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

import matplotlib.pyplot as plt



# Create scaler: scaler

scaler = StandardScaler()



# Create a PCA instance: pca

pca = PCA()



# Create pipeline: pipeline

pipeline = make_pipeline(scaler,pca)



# Fit the pipeline to 'samples'

pipeline.fit(train_valid_X)



# Plot the explained variances

features = range(pca.n_components_)

plt.bar(features, pca.explained_variance_)

plt.xlabel('PCA feature')

plt.ylabel('variance')

plt.xticks(features)

plt.show()
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import make_scorer, accuracy_score
model_type = []

train_acc = []

test_acc = []
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred_val = logreg.predict(X_val)

Y_pred_train = logreg.predict(X_train)



model_type.append('Logistic Regression')

print(accuracy_score(Y_train, Y_pred_train), accuracy_score(Y_val, Y_pred_val))

train_acc.append(accuracy_score(Y_train, Y_pred_train))

test_acc.append(accuracy_score(Y_val, Y_pred_val))
# Import necessary modules

from sklearn.model_selection import GridSearchCV



# Create the hyperparameter grid

c_space = np.logspace(-5, 8, 15)

param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}



# Instantiate the logistic regression classifier: logreg

logreg = LogisticRegression()



# Instantiate the GridSearchCV object: logreg_cv

logreg_cv = GridSearchCV(logreg, param_grid, cv=10)



# Fit it to the training data

logreg_cv.fit(X_train, Y_train)



# Print the optimal parameters and best score

print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))

print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

Y_pred_val = logreg_cv.predict(X_val)

Y_pred_train = logreg_cv.predict(X_train)

print(accuracy_score(Y_train, Y_pred_train), accuracy_score(Y_val, Y_pred_val))



model_type.append('GridSearchCV Logistic Regression')

train_acc.append(accuracy_score(Y_train, Y_pred_train))

test_acc.append(accuracy_score(Y_val, Y_pred_val))
# Import necessary modules

from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score



# Setup the array of alphas and lists to store scores

alpha_space = np.logspace(-5, 8, 15)

ridge_scores = []

ridge_scores_std = []



# Create a ridge regressor: ridge

ridge = Ridge(normalize=True)



# Compute scores over range of alphas

for alpha in alpha_space:



    # Specify the alpha value to use: ridge.alpha

    ridge.alpha = alpha

    

    # Perform 10-fold CV: ridge_cv_scores

    ridge_cv_scores = cross_val_score(ridge, X_train, Y_train, cv=10)

    

    print(alpha, np.mean(ridge_cv_scores), np.std(ridge_cv_scores))
# k-NN

for k in range (2, 10):

    knn = KNeighborsClassifier(n_neighbors = k)

    knn.fit(X_train, Y_train)

    Y_pred_val = knn.predict(X_val)

    Y_pred_train = knn.predict(X_train)

    print(k, accuracy_score(Y_train, Y_pred_train), accuracy_score(Y_val, Y_pred_val))
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred_val = knn.predict(X_val)

Y_pred_train = knn.predict(X_train)



model_type.append('k-NN')

train_acc.append(accuracy_score(Y_train, Y_pred_train))

test_acc.append(accuracy_score(Y_val, Y_pred_val))
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred_val = decision_tree.predict(X_val)

Y_pred_train = decision_tree.predict(X_train)

print(accuracy_score(Y_train, Y_pred_train), accuracy_score(Y_val, Y_pred_val))



model_type.append('Decision Tree')

train_acc.append(accuracy_score(Y_train, Y_pred_train))

test_acc.append(accuracy_score(Y_val, Y_pred_val))
from scipy.stats import randint

from sklearn.model_selection import RandomizedSearchCV



# Setup the parameters and distributions to sample from: param_dist

param_dist = {"max_depth": [3, 5],

              "max_features": randint(1, 6),

              "min_samples_leaf": randint(1, 6),

              "criterion": ["gini", "entropy"]}



# Instantiate a Decision Tree classifier: tree

tree = DecisionTreeClassifier()



# Instantiate the RandomizedSearchCV object: tree_cv

tree_cv = RandomizedSearchCV(tree,param_dist,cv=10)



# Fit it to the data

tree_cv.fit(X_train, Y_train)



# Print the tuned parameters and score

print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))

print("Best score is {}".format(tree_cv.best_score_))



Y_pred_val = tree_cv.predict(X_val)

Y_pred_train = tree_cv.predict(X_train)

print(accuracy_score(Y_train, Y_pred_train), accuracy_score(Y_val, Y_pred_val))



model_type.append('RandomizedSearch Decision Tree')

train_acc.append(accuracy_score(Y_train, Y_pred_train))

test_acc.append(accuracy_score(Y_val, Y_pred_val))
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_val)

Y_pred_train = random_forest.predict(X_train)

print(accuracy_score(Y_train, Y_pred_train), accuracy_score(Y_val, Y_pred_val))



model_type.append('Random Forest')

train_acc.append(accuracy_score(Y_train, Y_pred_train))

test_acc.append(accuracy_score(Y_val, Y_pred_val))
gbm_param_grid = {

    'n_estimators': np.arange(50,500,50),

}



# Instantiate the regressor: gbm

random_forest = RandomForestClassifier()



# Perform grid search: grid_mse

grid_random_forest = GridSearchCV(estimator=random_forest, param_grid=gbm_param_grid, scoring="neg_mean_squared_error",cv=10,verbose=1)



# Fit grid_mse to the data

grid_random_forest.fit(X_train, Y_train)



# Print the best parameters and lowest RMSE

print("Best parameters found: ", grid_random_forest.best_params_)

print("Lowest RMSE found: ", np.sqrt(np.abs(grid_random_forest.best_score_)))



Y_pred_val = grid_random_forest.predict(X_val)

Y_pred_train = grid_random_forest.predict(X_train)

print(accuracy_score(Y_train, Y_pred_train), accuracy_score(Y_val, Y_pred_val))



model_type.append('GridSearchCV Random Forest')

train_acc.append(accuracy_score(Y_train, Y_pred_train))

test_acc.append(accuracy_score(Y_val, Y_pred_val))
# This example uses the current build of XGBoost, from https://github.com/dmlc/xgboost

gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.05).fit(X_train, Y_train)

Y_pred_val = gbm.predict(X_val)

Y_pred_train = gbm.predict(X_train)

print(accuracy_score(Y_train, Y_pred_train), accuracy_score(Y_val, Y_pred_val))



model_type.append('XGBoost')

train_acc.append(accuracy_score(Y_train, Y_pred_train))

test_acc.append(accuracy_score(Y_val, Y_pred_val))
# Create the parameter grid: gbm_param_grid

gbm_param_grid = {

    'learning_rate': np.arange(.05, 0.5, .05),

    'n_estimators': np.arange(200,500,50),

    'max_depth': np.arange(2, 4, 1)

}



# Instantiate the regressor: gbm

gbm = xgb.XGBClassifier()



# Perform grid search: grid_mse

grid_mse = GridSearchCV(estimator=gbm, param_grid=gbm_param_grid, scoring="neg_mean_squared_error",cv=10,verbose=1)



# Fit grid_mse to the data

grid_mse.fit(X_train, Y_train)



# Print the best parameters and lowest RMSE

print("Best parameters found: ", grid_mse.best_params_)

print("Lowest RMSE found: ", np.sqrt(np.abs(grid_mse.best_score_)))
Y_pred_val = grid_mse.predict(X_val)

Y_pred_train = grid_mse.predict(X_train)

print(accuracy_score(Y_train, Y_pred_train), accuracy_score(Y_val, Y_pred_val))



model_type.append('GridSearchCV XGBoost')

train_acc.append(accuracy_score(Y_train, Y_pred_train))

test_acc.append(accuracy_score(Y_val, Y_pred_val))
summary = pd.DataFrame({ 'model_type' : model_type, 'train_acc': train_acc, 'test_acc': test_acc })

summary
predictions = grid_random_forest.predict(X_test)



output = pd.DataFrame({ 'PassengerId' : testID, 'Survived': predictions })

output.head()
output.to_csv('titanic-predictions.csv', index = False)