import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer



from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
test = pd.read_csv("../input/forest-cover-type-prediction/test.csv")

train = pd.read_csv("../input/forest-cover-type-prediction/train.csv")
train.head()
X_train = train.drop(['Id', 'Cover_Type'], axis=1)

y_train = train.Cover_Type

X_test = test.drop('Id', axis=1)

test_id = test.Id



print(X_train.shape)

print(X_test.shape)
print(list(zip(range(0,56), X_train.columns)))
num_features = X_train.columns

#cat_features = []
scaler = MinMaxScaler()

Xs_train = scaler.fit_transform(X_train)

Xs_test = scaler.transform(X_test)
lr_model = LogisticRegression(solver='liblinear')



param_grid = {

    'penalty': ['l1', 'l2'],

    'C': [10, 100, 1000],

}



np.random.seed(1)

grid_search = GridSearchCV(lr_model, param_grid, cv=5, refit='True', verbose=15, n_jobs=-1)

grid_search.fit(Xs_train, y_train)



print(grid_search.best_score_)

print(grid_search.best_params_)
rf_model = RandomForestClassifier(n_estimators=500)

    

param_grid = {

    'min_samples_leaf' : [2, 4, 8, 16],

    'max_depth' : [12, 16, 24]

}



np.random.seed(1)

grid_search = GridSearchCV(rf_model, param_grid, cv=5, refit='True', verbose=15, n_jobs=-1)

grid_search.fit(X_train, y_train)



print(grid_search.best_score_)

print(grid_search.best_params_)
test_pred = grid_search.predict(X_test)
for i in range(1,8):

    print(list(test_pred).count(i))

submission = pd.DataFrame({

    'Id':test_id,

    'Cover_Type':test_pred

})

submission.head()
submission.to_csv('my_submission.csv', index=False)