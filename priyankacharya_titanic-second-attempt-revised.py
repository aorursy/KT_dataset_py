# Import all required modules

import pandas as pd

import numpy as np

from sklearn import linear_model,model_selection,preprocessing
# Read training data

df_train = pd.read_csv("train.csv")

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

df_test = pd.read_csv("test.csv")

X_test = df_test[['Sex','Age','Pclass']]

# Preprocess the data for better fitting

X_test['Sex'] = X_test['Sex'].apply(lambda x:1 if x=="female" else 0)

X_test['Age'].fillna(value=np.mean(X_test['Age']),inplace=True)
# Predict test data

y_pred = grdsrch.predict(X_test)

y_pred
# Export to CSV for submission

pd.DataFrame({'PassengerId':df_test.PassengerId,'Survived':y_pred},columns=["PassengerId","Survived"]).to_csv("my_submission_2.csv",index=False)
# Normalize predictor variables

X_train_norm = preprocessing.normalize(X)

X_test_norm = preprocessing.normalize(X_test)
# Fit the model with normalized data

gs_fit = grdsrch.fit(X_train_norm,y)

gs_fit.best_score_,gs_fit.best_estimator_,gs_fit.cv_results_
# Predict the test data set based on normalized predictor variables

y_pred = grdsrch.predict(X_test_norm)

y_pred

# Export data to CSV for submission

pd.DataFrame({'PassengerId':df_test.PassengerId,'Survived':y_pred},columns=["PassengerId","Survived"]).to_csv("my_submission_3.csv",index=False)
# Make another field called Age Group by clubbing Age into AgeGroup

df_train['AgeGrp'] = pd.cut(df_train['Age'],bins=5,include_lowest=True).astype('category')

df_train['AgeGrpCode'] = df_train.AgeGrp.cat.codes



df_test['AgeGrp'] = pd.cut(df_train['Age'],bins=5,include_lowest=True).astype('category')

df_test['AgeGrpCode'] = df_train.AgeGrp.cat.codes
X.drop(columns='Age',axis=1)

X[['Sex','Pclass','AgeGrpCode']]
# Add additional predictor variables

X['AgeGrpCode'] = df_train['AgeGrpCode']

X_test['AgeGrpCode'] = df_test['AgeGrpCode']

X.drop(columns='Age')

X_test.drop(columns='Age')

# Normalize predictor variables

X_train_norm = preprocessing.normalize(X[['Sex','Pclass','AgeGrpCode']])

X_test_norm = preprocessing.normalize(X_test[['Sex','Pclass','AgeGrpCode']])

print(X_train_norm[0:10],X_test_norm[0:10])
# Fit the model with normalized data

C_vals = [1,100,1000,10000]

grdsrch = model_selection.GridSearchCV(estimator=model,param_grid={'C':C_vals},return_train_score=True,cv=5)

gs_fit = grdsrch.fit(X_train_norm,y)

gs_fit.best_score_,gs_fit.best_estimator_,gs_fit.cv_results_
# Predict the test data set based on normalized predictor variables

y_pred = grdsrch.predict(X_test_norm)

y_pred

# Export data to CSV for submission

pd.DataFrame({'PassengerId':df_test.PassengerId,'Survived':y_pred},columns=["PassengerId","Survived"]).to_csv("my_submission_4.csv",index=False)