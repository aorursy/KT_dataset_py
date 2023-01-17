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

from sklearn.ensemble import ExtraTreesClassifier



from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
test = pd.read_csv("../input/forest-cover-type-prediction/test.csv")

train = pd.read_csv("../input/forest-cover-type-prediction/train.csv")
print(train.head())
print(test.head())
print(train.shape)
print(test.shape)
print(train.dtypes)
print(np.unique(train.dtypes.values))
train.describe()

#no attribute is missing as count is consatant so can use all the attribute.soil types are onehot coading because all  are in 1 and 0.
train.isnull().sum()
X_train=train.drop(['Id', 'Cover_Type'],axis=1)

y_train=train.Cover_Type

X_test=test.drop('Id',axis=1)



test_id=test.Id   #Id and cover type is not required so drop it
print(X_train.shape)

print(X_test.shape)
print(list(zip(range(0,56),X_train.columns)))# To see all the columns of x_train with col no we used Zip() function inside a list which will return a list.
import seaborn as sns

sns.countplot(data=train,x=train['Cover_Type'])
sns.boxplot(x="Cover_Type", y="Elevation", data=train);
sns.boxplot(x="Cover_Type", y="Aspect",data=train);
scaler = MinMaxScaler()

Xs_train = scaler.fit_transform(X_train)

Xs_test = scaler.transform(X_test)

           #For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones.

lr_pipe = Pipeline(

    steps = [

        ('scaler', MinMaxScaler()),

        ('classifier', LogisticRegression(solver='lbfgs', n_jobs=-1))

    ]

)



lr_param_grid = {

    'classifier__C': [1, 10, 100,1000],

}





np.random.seed(1)

grid_search = GridSearchCV(lr_pipe, lr_param_grid, cv=5, refit='True')

grid_search.fit(X_train, y_train)



print(grid_search.best_score_)

print(grid_search.best_params_)

0.640079365079365





xrf_pipe = Pipeline(

    steps = [

        ('classifier', ExtraTreesClassifier(n_estimators=500,random_state=0))

    ]

)



param_grid = {

    'classifier__min_samples_leaf': [1,4,7,8],

    'classifier__max_depth': [31,34,36,39],



}



xrf_grid_search = GridSearchCV(xrf_pipe, param_grid, cv=5, refit='True', n_jobs=-1)

xrf_grid_search.fit(X_train, y_train)



print(xrf_grid_search.best_score_)

print(xrf_grid_search.best_params_)



#0.7837962962962962

#{'classifier__max_depth': 34, 'classifier__min_samples_leaf': 1}
xrf_model = xrf_grid_search.best_estimator_



cv_score = cross_val_score(xrf_model, X_train, y_train, cv = 5)

print(cv_score)

print( (cv_score.mean(), cv_score.std() * 2))
rf_pipe = Pipeline(

    steps = [

        ('classifier', RandomForestClassifier(n_estimators=500,random_state=0))

    ]

)



param_grid = {

    'classifier__min_samples_leaf': [1,4,7,8],

    'classifier__max_depth': [31,34,36,39],

}



np.random.seed(1)

rf_grid_search = GridSearchCV(rf_pipe, param_grid, cv=5, refit='True', n_jobs=-1)

rf_grid_search.fit(X_train, y_train)



print(rf_grid_search.best_score_)

print(rf_grid_search.best_params_)

rf_model = rf_grid_search.best_estimator_



cv_score = cross_val_score(rf_model, X_train, y_train, cv = 5)

print(cv_score)

print( (cv_score.mean(), cv_score.std() * 2))
xgd_pipe = Pipeline(

    steps = [

        ('classifier', XGBClassifier(n_estimators=50, subsample=0.5))

    ]

)



param_grid = {

    'classifier__learning_rate' : [0.45],

    'classifier__min_samples_split' : [8, 16, 32],

    'classifier__min_samples_leaf' : [2],

    'classifier__max_depth': [15]

    

}



np.random.seed(1)

xgd_grid_search = GridSearchCV(xgd_pipe, param_grid, cv=5,

                              refit='True', verbose = 10, n_jobs=-1)

xgd_grid_search.fit(X_train, y_train)



print(xgd_grid_search.best_score_)

print(xgd_grid_search.best_params_)

xgd_model = xgd_grid_search.best_estimator_



cv_score = cross_val_score(xgd_model, X_train, y_train, cv = 5)

print(cv_score)

print( (cv_score.mean(), cv_score.std() * 2))
test_pred = xrf_grid_search.predict(X_test)
submission = pd.DataFrame({

    'Id':test_id,

    'Cover_Type':test_pred

})

submission.head()
submission.to_csv('my_submission.csv', index=False)