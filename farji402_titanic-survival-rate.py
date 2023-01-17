import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



#reading train data

df = pd.read_csv('../input/titanic/train.csv')



"""print(df.isnull().sum())"""

#reading test data

df_test = pd.read_csv('../input/titanic/test.csv')



#treating test data

df_test.drop(columns= ['Cabin','Age', 'Name', 'Ticket'], inplace= True)

df_test['Sex'] = df_test['Sex'].astype('category')

df_test['Embarked'] = df_test['Embarked'].astype('category')

df_test_num = pd.get_dummies(df_test, drop_first= True)

df_test_num.fillna(df_test_num['Fare'].mean(), inplace= True)







#treating train data

df.drop(columns= ['Cabin','Age', 'Name', 'Ticket'], inplace= True)

df = df.dropna()

df['Sex'] = df['Sex'].astype('category')

df['Embarked'] = df['Embarked'].astype('category')

df_num = pd.get_dummies(df, drop_first= True)







#Model 

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

from sklearn.pipeline import Pipeline 

from sklearn.svm import SVC

import xgboost as xgb



X = df_num.drop(columns= ['Survived'])

y = df_num['Survived']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.4, random_state= 20, stratify= y)





"""paras= {'n_estimators':np.arange(100, 700, 50)}

gbt_clf = GradientBoostingClassifier(max_depth=1, random_state=20, subsample=0.8,max_features=0.2)

gbt = GridSearchCV(gbt_clf, paras, cv=5)

gbt.fit(X, y)

#y_pred_gbt = gbt.predict_proba(X_test)[:,1]

#score_gbt = roc_auc_score(y_test, y_pred_gbt)

#print('ROC AUC score of gbt: {:.2f}'.format(score_gbt))

#print('Accuracy on train data: {}'.format(gbt.score(X_train, y_train)))

#print('Accuracy on test data: {}'.format(gbt.score(X_test, y_test)))"""







params = { 'max_depth':np.arange(1,10,1),

         'learning_rate':np.arange(0.05, 1.05, 0.05),

         'n_estimators':[200],

         'subsample':np.arange(0.05, 1.05, 0.05),

         'colsample_bytree': np.arange(0.05,1,0.05)}

searcher = RandomizedSearchCV(estimator= xgb.XGBClassifier(), param_distributions= params,

                             n_iter= 50, scoring= 'accuracy', cv= 4, verbose= 1, n_jobs= -1)

searcher.fit(X, y)

print("best score: ", searcher.best_score_)





prediction = searcher.predict(df_test_num)



pred_dict = { 'PassengerId': df_test_num['PassengerId'].values, 'Survived': prediction}



df_pred = pd.DataFrame(pred_dict)

df_pred = df_pred.sort_values('PassengerId')





print('\n', df_pred.head())

df_pred.to_csv('submission.csv', index= False )