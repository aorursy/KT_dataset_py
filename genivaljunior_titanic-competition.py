# Import modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
# Importing training dataset

dftrain = pd.read_csv('../input/train.csv')

dftrain.head(3)
# Checking data types

dftrain.dtypes
# Adjusting some data, in order to make analysis more feasible

# Sex: let's use 1 for 'male' and 2 for 'female'

# Embarked: Let's use 1 for 'Cherbourg', 2 for 'Queenstown' and 3 for 'Southampton'



dftrain['Sex2'] = dftrain['Sex'].apply(lambda x: 1 if x == 'male' else 2)

dftrain['Embarked2'] = dftrain['Embarked'].apply(lambda x: 1 if x == 'C' else 2 if x == 'Q' else 3)
# Statistical summary 

dftrain.iloc[:, 1:].describe()
# Correlation Matrix

correlations = dftrain.iloc[:, 1:].corr()

colunas = list(correlations.columns)



# Plot

fig = plt.figure()

ax = fig.add_subplot(111)

cax = ax.matshow(correlations, vmin=-1, vmax=1)

fig.colorbar(cax)

ticks = np.arange(0, 8, 1)

ax.set_xticks(ticks)

ax.set_yticks(ticks)

ax.set_xticklabels(colunas)

ax.set_yticklabels(colunas)

plt.show()



correlations
# View on Seaborn



# Pairplot

dftrain['Age'] = dftrain["Age"].replace(np.nan, 0)

df1 = dftrain.loc[:, colunas]

sns.pairplot(df1)

plt.show()
# Features Histograms

df1.hist(layout=(3,3), sharex=False)

plt.show()
# Importing sklearn modules

from sklearn import model_selection

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

#from sklearn.cluster import KMeans

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
# Preparing dataset

dtrain = df1[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex2', 'Embarked2', 'Survived']]

array = dtrain.values



X = array[:, 0:7]

Y = array[:, 7]
# Creating a list of classification models

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('NB', GaussianNB()))

models.append(('DTC', DecisionTreeClassifier()))

models.append(('SVC', SVC()))

models.append(('XGB', GradientBoostingClassifier()))

models.append(('RDF', RandomForestClassifier()))

#models.append(('KME', KMeans()))



# Applying models on training dataset and evaluating their accuracy

results = []

names = []



for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=7)

    cross_val_result = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')

    

    results.append(cross_val_result)

    names.append(name)

    

    text = '%s: %.3f (%.3f)' % (name, cross_val_result.mean(), cross_val_result.std())

    print(text)



fig = plt.figure()

fig.suptitle('Comparing algorithms')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
# Now, we repeat the process, but standardizating the features scales, in order to improve accuracy

# Let's use the pipeline resource to leverage our job

pipelines = []

pipelines.append(('Scl-LR', Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression())])))

pipelines.append(('Scl-LDA', Pipeline([('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])))

pipelines.append(('Scl-KNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])))

pipelines.append(('Scl-NB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])))

pipelines.append(('Scl-DTC', Pipeline([('Scaler', StandardScaler()), ('DTC', DecisionTreeClassifier())])))

pipelines.append(('Scl-SVC', Pipeline([('Scaler', StandardScaler()), ('SVC', SVC())])))

pipelines.append(('Scl-XGB', Pipeline([('Scaler', StandardScaler()), ('XGB', GradientBoostingClassifier())])))

pipelines.append(('Scl-RDF', Pipeline([('Scaler', StandardScaler()), ('RDF', RandomForestClassifier())])))

#pipelines.append(('Scl-KME', Pipeline([('Scaler', StandardScaler()), ('KME', KMeans())])))



# Applying models on training dataset and evaluating their accuracy

results = []

names = []



for name, model in pipelines:

    kfold = model_selection.KFold(n_splits=10, random_state=7)

    cross_val_result = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')

    

    results.append(cross_val_result)

    names.append(name)

    

    text = '%s: %.3f (%.3f)' % (name, cross_val_result.mean(), cross_val_result.std())

    print(text)



fig = plt.figure()

fig.suptitle('Comparing algorithms')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
# Using GridSearch ro optimize XGB parameters

# Defining Scale

scaler = StandardScaler().fit(X)

rescaledX = scaler.transform(X)



# Defining values for parameters n_estimator and learning_rate:

n_estim = [100, 500, 1000]

l_rate = [0.1, 0.5, 1.0]

m_dep = [1, 2, 3]

values_grid = dict(n_estimators = n_estim, learning_rate = l_rate, max_depth = m_dep)



# Creating model

xgb = GradientBoostingClassifier()



# Testing parameters combinations

kfold = model_selection.KFold(n_splits=10, random_state=7)



grid = model_selection.GridSearchCV(estimator=xgb, param_grid=values_grid, 

                                    cv = kfold, scoring='accuracy', return_train_score=True)

grid_result = grid.fit(rescaledX, Y)



# Evaluating model tunning

mean_score = [mean for mean in grid_result.cv_results_['mean_test_score']]

std_score = [mean for mean in grid_result.cv_results_['std_test_score']]

params_score = [mean for mean in grid_result.cv_results_['params']]



# Output

print('Best accuracy: %.4f using %s' % (grid_result.best_score_, grid_result.best_params_), '\n')

i = 0

for param in params_score:

    print('%.4f (%.4f) with %r' % (mean_score[i], std_score[i], param))

    i = i + 1;

# Using GridSearch ro optimize SVC parameters

# Defining Scale

scaler = StandardScaler().fit(X)

rescaledX = scaler.transform(X)



# Defining values for parameters n_estimator and learning_rate:

c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]

kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']

values_grid = dict(C = c_values, kernel = kernel_values)



# Creating model

svc = SVC()



# Testing parameters combinations

kfold = model_selection.KFold(n_splits=10, random_state=7)



grid = model_selection.GridSearchCV(estimator=svc, param_grid=values_grid, 

                                    cv = kfold, scoring='accuracy', return_train_score=True)

grid_result = grid.fit(rescaledX, Y)



# Evaluating model tunning

mean_score = [mean for mean in grid_result.cv_results_['mean_test_score']]

std_score = [mean for mean in grid_result.cv_results_['std_test_score']]

params_score = [mean for mean in grid_result.cv_results_['params']]



# Output

print('Best accuracy: %.4f using %s' % (grid_result.best_score_, grid_result.best_params_))

i = 0

for param in params_score:

    print('%.4f (%.4f) with %r' % (mean_score[i], std_score[i], param))

    i = i + 1;
# Fitting the selected model

xgb = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.1, max_depth=2)

xgb.fit(rescaledX, Y)
# Importing test dataset

dftest = pd.read_csv('../input/test.csv')

dftest.head()
# Preparing dataset, just like it was done on the training fase

dftest['Sex2'] = dftest['Sex'].apply(lambda x: 1 if x == 'male' else 2)

dftest['Embarked2'] = dftest['Embarked'].apply(lambda x: 1 if x == 'C' else 2 if x == 'Q' else 3)

dftest['Age'] = dftest["Age"].replace(np.nan, 0)

dftest['Fare'] = dftest["Fare"].replace(np.nan, 0)



dtest = dftest[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex2', 'Embarked2']]

X_tst = dtest.values

X_tst
# Adjusting scale

scalerval = StandardScaler().fit(X_tst)

rescaledvalX = scalerval.transform(X_tst)



# Applying the model (XGBoost)

Survived_XGB_val = xgb.predict(rescaledvalX)
# Creating the arqchive with the predictions

dftest['Survived'] = Survived_XGB_val

dftest['Survived'] = dftest['Survived'].apply(lambda x: int(x))

output = dftest[['PassengerId', 'Survived']]

output.head()
output.groupby('Survived').count()