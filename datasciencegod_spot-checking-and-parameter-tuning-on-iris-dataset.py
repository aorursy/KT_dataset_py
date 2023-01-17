# Imports

from sklearn.datasets import load_iris

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import cross_val_score

from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.grid_search import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns
# Loading Dataset

iris = load_iris()
# Creating features and class

x = iris.data

y = iris.target
# Creating Models dictionary

models = {'SVM':SVC(), 'logreg': LogisticRegression(), 'RandomForestClassifier': RandomForestClassifier(), 

          'Naive-Bayes Gauss': GaussianNB(), 'Naive-Bayes Multi': MultinomialNB(), 'Decision tree': DecisionTreeClassifier(),

         'Knn': KNeighborsClassifier(n_neighbors=20), 'GradientBoost': GradientBoostingClassifier(),

          'AdaBoost':AdaBoostClassifier(base_estimator= DecisionTreeClassifier())}
# Scoring

for key in models:

    score = cross_val_score(models[key], x, y, cv=13, scoring='accuracy').mean()

    print('{} scored: {:.1f}'.format(key, score*100))
# Taking SVM ahead and improving it
# Finding optimized paramters: Defining the grid to be searched



param_grid = dict(kernel = ['linear', 'rbf', 'poly', 'sigmoid'], decision_function_shape=['ovr', 'ovo'])

svm_model = SVC()
# Initializing the grid

grid = GridSearchCV(svm_model, param_grid, cv =10, scoring='accuracy', n_jobs=-1)
# Fitting the grid with data

grid.fit(x, y)
# Getting best estimator settings

grid.best_estimator_
# Optimized parameters from GridSearchCV

grid.best_params_
# Best score obtained while searching the grid

grid.best_score_
grid.grid_scores_
# Initializing SVM model with optimized parameter values

svmc = SVC(decision_function_shape='ovr', kernel='rbf',)

# Ovr is used because it is a multi class problem.
# Finding new score

new_scoresvm = cross_val_score(svmc, x, y, cv = 10, scoring='accuracy').mean()

print(new_scoresvm)
# Percentage improvement using optimized parameters

print('Improvement by {:0.2}% '.format(((new_scoresvm-0.974)/0.974)*100))
# Taking KNN and trying to improve the accuracy

knn = KNeighborsClassifier()



cross_val_score(knn, x, y, cv=13, scoring='accuracy').mean()
list(range(5,26))
# Defining parameter grid for the search

param_gridKNN = dict(n_neighbors = list(range(5,26)), weights = ['uniform', 'distance'], 

                     algorithm =['auto', 'ball_tree', 'kd_tree', 'brute'],

                    leaf_size = [10,20,30,40,50])



# Initializing the grid

gridknn = GridSearchCV(knn, param_gridKNN, cv =10, scoring='accuracy', n_jobs=-1)
# Fitting the grid with data

gridknn.fit(x, y)
# Getting the best estimator

gridknn.best_estimator_
# Getting the best parameters

gridknn.best_params_
# Getting the best score

gridknn.best_score_
# Initializing the KNN model with new Hyperparamters

knn = KNeighborsClassifier(algorithm='auto', leaf_size=10, metric='minkowski',

           metric_params=None, n_jobs=1, n_neighbors=13, p=2,

           weights='uniform')
# New score from optimized knn

new_scoreknn = cross_val_score(knn, x, y, cv = 10, scoring='accuracy').mean()

new_scoreknn
# Percentage improvement using optimized parameters

print('Improvement by {:0.2}% '.format(((new_scoreknn-0.974)/0.974)*100))
# Tuning GradientBoost

gb = GradientBoostingClassifier()

cross_val_score(gb, x, y, cv=13, scoring='accuracy').mean()
# Setting up the param_grid

param_gridgb = dict(n_estimators = [50,100,150,200,250,300], max_depth = [2,3,4,5,6], learning_rate = [0.001, 0.01,0.1,1])



# Initializing the grid

gridgb = GridSearchCV(gb, param_gridgb, cv =10, scoring='accuracy', n_jobs=-1)

# fitting the grid

gridgb.fit(x,y)
gridgb.best_estimator_
gridgb.best_params_
new_scoregb = gridgb.best_score_