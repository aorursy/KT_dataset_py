# Import all required modules

import pandas as pd

import numpy as np

from sklearn import linear_model,model_selection,preprocessing
# Read training data

df_train = pd.read_csv("../input/titanic/train.csv")

X = df_train[['Sex','Age','Pclass']]

y = df_train['Survived']
# Preprocess the data for better fitting

X['Sex'] = X['Sex'].apply(lambda x:1 if x=="female" else 0)

X['Age'].fillna(value=np.mean(X['Age']),inplace=True)

X
# Apply logistic regression algorithm

model = linear_model.LogisticRegression(solver='lbfgs')

model.fit(X,y)

model.classes_,model.coef_,model.intercept_

model.score(X,y)
C_vals = [1e-3,0.1,1,100,1000,1e5]

grdsrch = model_selection.GridSearchCV(estimator=model,param_grid={'C':C_vals},return_train_score=True,cv=5)

gs_fit = grdsrch.fit(X,y)

gs_fit.best_score_,gs_fit.best_params_,gs_fit.cv_results_
# Read the test data

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

X_test = df_test[['Sex','Age','Pclass']]

# Preprocess the data for better fitting

X_test['Sex'] = X_test['Sex'].apply(lambda x:1 if x=="female" else 0)

X_test['Age'].fillna(value=np.mean(X_test['Age']),inplace=True)
#X_test

y_pred = grdsrch.predict(X_test)

y_pred
pd.DataFrame({'PassengerId':df_test.PassengerId,'Survived':y_pred},columns=["PassengerId","Survived"]).to_csv("my_submission_2.csv",index=False)
X_train_norm = preprocessing.normalize(X)

X_test_norm = preprocessing.normalize(X_test)
gs_fit = grdsrch.fit(X_train_norm,y)

gs_fit.best_score_,gs_fit.best_estimator_,gs_fit.cv_results_
y_pred = grdsrch.predict(X_test_norm)

y_pred

pd.DataFrame({'PassengerId':df_test.PassengerId,'Survived':y_pred},columns=["PassengerId","Survived"]).to_csv("my_submission_2.csv",index=False)