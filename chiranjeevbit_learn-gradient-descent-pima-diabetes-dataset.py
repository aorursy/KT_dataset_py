#importing the classes and functions we intend to use in this tutorial.
from numpy import loadtxt
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from xgboost import plot_importance
# load data
dataset = pd.read_csv('../input/diabetes.csv')
dataset.head()
# split data into X and y
X = dataset.iloc[:,0:8]
Y = dataset.iloc[:,8]

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)
print(model)
# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print(model.feature_importances_)
#The XGBoost library provides a built-in function to plot features ordered by their importance.
from xgboost import plot_importance
#The function is called plot_importance() and can be used as follows:

plot_importance(model)
pyplot.show()

#The scikit-learn framework provides the capability to search combinations of parameters.

#This capability is provided in the GridSearchCV class and can be used to discover the best way to configure the model for top performance on your problem.


#The number and size of trees (n_estimators and max_depth).
#The learning rate and number of trees (learning_rate and n_estimators).
#The row and column subsampling rates (subsample, colsample_bytree and colsample_bylevel).
#Below is a full example of tuning just the learning_rate on the Pima Indians Onset of Diabetes dataset.


# Tune learning_rate
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# grid search
model = XGBClassifier()
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
param_grid = dict(learning_rate=learning_rate)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
grid_result = grid_search.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

