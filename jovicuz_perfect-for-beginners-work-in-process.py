import pandas as pd

import numpy as np

df=pd.read_csv('../input/diabetes.csv')
missing_data=df.isnull()

missing_data.head(5)
missing_data.sum()
df.head(10)
df.shape
df.info()
#On classification problems you need to know how balanced the class values are.( This is an example)

df.groupby('Outcome').size() 
# We can analyze all the data set 

df.describe()
#We can analyze any colums separate

#df['MODELYEAR'].describe()
# When we have categorical values in the data set, we can create a table and sumarize it

#df.describe(include=['O'])
df.corr()
corr_matrix= df.corr()
#To check a correlation with our target



corr_matrix['Outcome'].sort_values(ascending=False)

import seaborn as sns
sns.heatmap(df.corr(), vmin=-1, vmax=1.0, annot=True)
df.skew()
from matplotlib import pyplot as plt

df.hist(bins=10, figsize=(20,15))

plt.show()
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,15))

plt.show()
from pandas.tools.plotting import scatter_matrix



scatter_matrix(df,figsize=(20,20))

plt.show()
missing_data= df.isnull()
missing_data.head(5)
missing_data.sum()
for column in missing_data.columns.values.tolist():

    print(column)

    print(missing_data[column].value_counts())

    print("--------------------------------")
#Finding the porcentage of  missing data

round(((missing_data.sum()/len(missing_data))*100), 4)
# q = df.quantile(0.99)

#df [df > q]
#Lets check the types 

df.dtypes

#df = pd.get_dummies(df, prefix_sep='_', drop_first=True)
df.shape
X=df.drop('Outcome',axis=1)

y=df['Outcome']
import pandas as pd

import numpy as np

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

#apply SelectKBest class to extract top 10 best features

bestfeatures = SelectKBest(score_func=chi2, k='all')

fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(10,'Score'))  #print 10 best features
import pandas as pd

import numpy as np

from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X,y)

print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=X.columns)

feat_importances.nlargest(10).plot(kind='barh')

plt.show()
df.columns
#X_train=

#y_train=

#X_test=



X=df.drop(['Outcome'], axis=1).values

y=df['Outcome'].values

#from sklearn import preprocessing

#X= preprocessing.StandardScaler().fit(X).transform(X.astype(float))
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

rescaledX = scaler.fit_transform(X)

#X_test= preprocessing.StandardScaler().fit(X_test).transform(X_test.astype(float))


#X_train.shape, y_train.shape, X_test.shape
#Load Libraries

from sklearn.linear_model import LinearRegression 

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from xgboost.sklearn import XGBClassifier  

from xgboost.sklearn import XGBRegressor

from sklearn import preprocessing

from scipy.stats import uniform

from sklearn.linear_model import Ridge

from sklearn.model_selection import RandomizedSearchCV
#Linear Regression 

kfold = KFold(n_splits=10, random_state=42,)

lin_reg= LinearRegression()

results_linreg= cross_val_score(lin_reg, X, y, cv=kfold)

print('Estimate accuracy',results_linreg.mean())


# Logistic Regression

kfold = KFold(n_splits=10, random_state=42)

logreg = LogisticRegression(solver='lbfgs',max_iter=10000)

results_logreg = cross_val_score(logreg, X, y, cv=kfold,scoring='accuracy')

print('Estimate accuracy',results_logreg.mean())
# Support Vector Machines

kfold = KFold(n_splits=10, random_state=42)

svc = SVC(C=0.01, cache_size=200, class_weight=None, coef0=0.0,

   decision_function_shape='ovr', degree=3, gamma=1, kernel='linear',

   max_iter=-1, probability=False, random_state=None, shrinking=True,

   tol=0.001, verbose=False)

results_svc = cross_val_score(svc, X, y, cv=kfold,scoring='accuracy')

print('Estimate accuracy',results_svc.mean())
kfold = KFold(n_splits=10, random_state=42)

knn = KNeighborsClassifier(n_neighbors = 3)

results_knn = cross_val_score(knn, X, y, cv=kfold)

print('Estimate accuracy',results_knn.mean())
# Gaussian Naive Bayes

kfold = KFold(n_splits=10, random_state=42)

gaussian = GaussianNB()

results_gaussian = cross_val_score(gaussian, X, y, cv=kfold)

print('Estimate accuracy',results_gaussian.mean())
# Perceptron

kfold = KFold(n_splits=10, random_state=42)

perceptron = Perceptron(max_iter=1000,tol=1e-3)

results_perceptron = cross_val_score(perceptron, X, y, cv=kfold,scoring='accuracy')

print('Estimate accuracy',results_perceptron.mean())
# Linear SVC

kfold = KFold(n_splits=10, random_state=42)

linear_svc = LinearSVC(max_iter=1000)

results_linearsvc= cross_val_score(linear_svc, X, y, cv=kfold, scoring='accuracy')

print('Estimate accuracy',results_linearsvc.mean())
# Stochastic Gradient Descent

kfold = KFold(n_splits=10, random_state=42)

sgd = SGDClassifier(max_iter=1000,tol=1e-3)

results_sgd = cross_val_score(sgd, X, y, cv=kfold, scoring='accuracy')

print('Estimate accuracy',results_sgd.mean())
# Decision Tree

kfold = KFold(n_splits=10, random_state=42)

decision_tree = DecisionTreeClassifier()

results_decisiontree = cross_val_score(decision_tree, X, y, cv=kfold,scoring='accuracy')

print('Estimate accuracy',results_decisiontree.mean())
# Random Forest

kfold = KFold(n_splits=10, random_state=42)

random_forest = RandomForestClassifier(n_estimators=100)

results_randomforest = cross_val_score(decision_tree, X, y, cv=kfold,scoring='accuracy')

print('Estimate accuracy',results_randomforest.mean())
#Linear Discriminant Analysis

kfold = KFold(n_splits=10, random_state=42)

clf = LinearDiscriminantAnalysis()

results_clf = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')

print('Estimate accuracy',results_clf.mean())
# Ada Boost Classifier

kfold = KFold(n_splits=10, random_state=42)

AB = AdaBoostClassifier()

results_AB = cross_val_score(AB, X, y, cv=kfold, scoring='accuracy')

print('Estimate accuracy',results_AB.mean())



#AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm=’SAMME.R’, random_state=None)

# Gradient Boosting Classifier

kfold = KFold(n_splits=10, random_state=42)

GBC = GradientBoostingClassifier()

results_GBC = cross_val_score(GBC, X, y, cv=kfold, scoring='accuracy')

print('Estimate accuracy',results_GBC.mean())

#GradientBoostingClassifier(loss=’deviance’, learning_rate=0.1, n_estimators=100, subsample=1.0, criterion=’friedman_mse’, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort=’auto’, validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
#ExtraTreesClassifier

kfold = KFold(n_splits=10, random_state=42)

ETC=ExtraTreesClassifier(n_estimators=100)

results_ETC = cross_val_score(ETC, X, y, cv=kfold, scoring='accuracy')

print('Estimate accuracy',results_ETC.mean())

#ExtraTreesClassifier(n_estimators=’warn’, criterion=’gini’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)
#XGBClassifier(objective

xgbs = XGBClassifier(objective="binary:logistic", random_state=42)

results_xgbs = cross_val_score(xgbs, X, y, cv=kfold, scoring='accuracy')

print('Estimate accuracy',results_xgbs.mean())
models = pd.DataFrame({

    'Model': ['Linear Regression','Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree','Linear Discriminant Analysis','Ada Boost Classifier','Gradient Boosting Classifier','Extra TreesClassifier','XGB Classifier'],

    'Score': [results_linreg.mean(),results_logreg.mean(),results_svc.mean(),results_knn.mean(),results_gaussian.mean(),results_perceptron.mean(),results_linearsvc.mean(),results_sgd.mean(),results_decisiontree.mean(),results_randomforest.mean(),results_clf.mean(),results_AB.mean(),results_GBC.mean(),results_ETC.mean(),results_xgbs.mean()]})

models.sort_values(by='Score', ascending=False)
def svc_param_selection(X, y, nfolds):

    Cs = [0.001, 0.01, 0.1, 1, 10]

    gammas = [0.001, 0.01, 0.1, 1]

    param_grid = {'C': Cs, 'gamma' : gammas}

    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)

    grid_search.fit(X, y)

    grid_search.best_params_

    return grid_search.best_score_, grid_search.best_params_ ,grid_search.best_estimator_
svc_param_selection(X, y, 20)
def svc_param_selection2(X,y,nfolds):

    Cs = [0.001, 0.01, 0.1, 1, 10]

    gammas = [0.001, 0.01, 0.1, 1]

    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    param_dist = {'C': Cs, 'gamma' : gammas, 'kernel': kernels}

    rand = RandomizedSearchCV(SVC(), param_dist, cv=nfolds, scoring='accuracy', n_iter=10, random_state=42)

    rand.fit(X,y)

    rand.best_score_

    rand.best_params_

    rand.best_estimator_

    return  rand.best_score_, rand.best_params_ ,rand.best_estimator_
svc_param_selection2(X,y,20)