import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import StandardScaler

from subprocess import check_output
test = pd.read_csv("../input/data-science-london-scikit-learn/test.csv",header = None)

train = pd.read_csv("../input/data-science-london-scikit-learn/train.csv",header = None)

trainLabels = pd.read_csv("../input/data-science-london-scikit-learn/trainLabels.csv",header = None)
train.head(10)
train.info()
trainLabels = np.ravel(trainLabels)

trainLabels.shape

X_train, X_test, Y_train, Y_test = train_test_split(train, trainLabels, test_size = 0.25, random_state = 0)

print(X_train.shape)
scaled_logistic_pipe = Pipeline(steps = [('scaler', StandardScaler()),("classifier", LogisticRegression())])

scaled_logistic_param = {"classifier__solver": ["liblinear"], "classifier__penalty":['l1', 'l2']}

log_grid = GridSearchCV(scaled_logistic_pipe, scaled_logistic_param, cv = 10)

log_grid.fit(train, trainLabels)

log_model = log_grid.best_estimator_

print("Training score: ", log_model.score(X_train, Y_train))

print("Best Parameter: ", log_grid.best_params_)

print("Cross Validation Score:", log_grid.best_score_)
unscaled_knn_pipe = Pipeline(

steps = [

        ('classifier', KNeighborsClassifier())

    ]

)

 

unscaled_knn_param_grid = {

    'classifier__n_neighbors': range(1,10),

    'classifier__p': [1,2,3]



}



np.random.seed(1)



unscaled_knn_grid_search = GridSearchCV(unscaled_knn_pipe, unscaled_knn_param_grid, cv=10, refit='True')



unscaled_knn_grid_search.fit(X_train, Y_train)



unscaled_knn_model = unscaled_knn_grid_search.best_estimator_

 

print('Cross Validation Score:', unscaled_knn_grid_search.best_score_)



print('Best Hyperparameters:  ', unscaled_knn_grid_search.best_params_)



print('Training Accuracy:     ', unscaled_knn_model.score(X_train, Y_train))
scaled_knn_pipe = Pipeline(

steps = [

        ('scaler', StandardScaler()),

        ('classifier', KNeighborsClassifier())

    ]

)

 

scaled_knn_param_grid = {

    'classifier__n_neighbors': range(1,10),

    'classifier__p': [1,2,3]



}



np.random.seed(1)



scaled_knn_grid_search = GridSearchCV(scaled_knn_pipe, scaled_knn_param_grid, cv=10, refit='True')



scaled_knn_grid_search.fit(X_train, Y_train)



scaled_knn_model = scaled_knn_grid_search.best_estimator_

 

print('Cross Validation Score:', scaled_knn_grid_search.best_score_)



print('Best Hyperparameters:  ', scaled_knn_grid_search.best_params_)



print('Training Accuracy:     ', scaled_knn_model.score(X_train, Y_train))



unscaled_decision_pipe = Pipeline(

steps = [

        ('classifier', DecisionTreeClassifier())

    ]

)

 

unscaled_decision_param_grid = {

    'classifier__max_depth':[7,8,9,15],

    'classifier__random_state': [1,2,17]

}



np.random.seed(1)



unscaled_decision_grid_search = GridSearchCV(unscaled_decision_pipe, unscaled_decision_param_grid, cv=10, refit='True')



unscaled_decision_grid_search.fit(X_train, Y_train)



unscaled_decision_model = unscaled_decision_grid_search.best_estimator_

 

print('Cross Validation Score:', unscaled_decision_grid_search.best_score_)



print('Best Hyperparameters:  ', unscaled_decision_grid_search.best_params_)



print('Training Accuracy:     ', unscaled_decision_model.score(X_train, Y_train))



unscaled_randomforest_pipe = Pipeline(

steps = [

        ('classifier', RandomForestClassifier())

    ]

)

 

unscaled_randomforest_param_grid = {

    'classifier__min_samples_leaf': [2,3,8],

    'classifier__max_depth':[32,64,72]

}



unscaled_randomforest_param_grid = GridSearchCV(unscaled_randomforest_pipe, unscaled_randomforest_param_grid, cv = 10)



unscaled_randomforest_param_grid.fit(train,trainLabels)



unscaled_randomforest_model = unscaled_randomforest_param_grid.best_estimator_



print("Best parameter", unscaled_randomforest_param_grid.best_params_)



print("Best training accuracy", unscaled_randomforest_model.score(X_train, Y_train))



print("Best cross validation score", unscaled_randomforest_param_grid.best_score_)
final_test = unscaled_knn_model.predict(test)

final_test.shape
submission = pd.DataFrame(final_test)

print(submission.shape)

submission.columns = ['Solution']

submission['Id'] = np.arange(1,submission.shape[0]+1)

submission = submission[['Id', 'Solution']]

submission
filename = 'Scikit-Decision-Unscaled.csv'



submission.to_csv(filename,index=False)



print('Saved file: ' + filename)
print(check_output(["ls", "../working"]).decode("utf8"))