import numpy as np

import pandas as pd

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
import pandas as pd

test = pd.read_csv("../input/data-science-london-scikit-learn/test.csv",header = None)

train = pd.read_csv("../input/data-science-london-scikit-learn/train.csv",header = None)

trainLabels = pd.read_csv("../input/data-science-london-scikit-learn/trainLabels.csv", header = None)
train.head(10)
trainLabels = np.ravel(trainLabels)

trainLabels.shape
test.shape
X_train, X_valid, Y_train, Y_valid = train_test_split(train, trainLabels, test_size = 0.3, random_state = 1, 

                                                      stratify = trainLabels )

print(X_train.shape)
log_pipe = Pipeline(steps = [("classifier", LogisticRegression(solver='lbfgs', penalty='none'))])

log_param = {"classifier__solver": ["liblinear"], "classifier__penalty":['l1', 'l2'], 

             "classifier__C":[0.0001, 0.0005, 0.1, 0.5, 1.0, 5.0, 10]}

log_grid = GridSearchCV(log_pipe, log_param, cv = 10)

log_grid.fit(train, trainLabels)

log_model = log_grid.best_estimator_

print("Training score: ", log_model.score(X_train, Y_train))

print("Best Parameter: ", log_grid.best_params_)

print("Cross Validation Score:", log_grid.best_score_)
tree_pipe = Pipeline(steps = [ ("classifier", RandomForestClassifier(n_estimators = 100))])

tree_param = {"classifier__min_samples_leaf" : [2,3,8,16,32], "classifier__max_depth" :[6,12,16,32,64]}

tree_grid = GridSearchCV(tree_pipe, tree_param, cv = 10)

tree_grid.fit(train,trainLabels)

tree_model = tree_grid.best_estimator_

print("Best parameter", tree_grid.best_params_)

print("Best training accuracy", tree_model.score(X_train, Y_train))

print("Best cross validation score", tree_grid.best_score_)
knn_pipe = Pipeline(steps = [("classifier", KNeighborsClassifier())])

knn_param = {"classifier__n_neighbors" : range(1, 11), "classifier__p" : range(1, 3)}

np.random.seed(1)

grid_knn = GridSearchCV(knn_pipe, knn_param, cv = 10, refit = "TRUE")

grid_knn.fit(train, trainLabels)

knn_model = grid_knn.best_estimator_

print('Cross Validation Score:', grid_knn.best_score_)

print('Best Hyperparameters:  ', grid_knn.best_params_)

print('Training Accuracy:     ', knn_model.score(X_train, Y_train))
pred_test = grid_knn.predict(test)

pred_test[:5]

pred_test.shape
submission = pd.DataFrame(pred_test)

submission.columns = ['Solution']

submission['Id'] = np.arange(1,submission.shape[0]+1)

submission = submission[['Id', 'Solution']]

submission.head()
filename = 'London1995.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)