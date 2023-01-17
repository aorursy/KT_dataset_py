# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df=pd.read_csv("/kaggle/input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv")
train_df.head()
train_df.info()
#null_columns=train_df.columns[train_df.isnull().any()]

#train_df[null_columns].isnull().sum()



train_df.isnull().sum()
#When inplace = True is used, it performs operation on data and nothing is returned.

#df.some_operation(inplace=True)

#When inplace=False is used, it performs operation on data and returns a new copy of data.

#df = df.an_operation(inplace=False)

train_df['Gender'].fillna("No Gender Data",inplace=True)

train_df['Married'].fillna("No Married Data",inplace=True)

train_df['Dependents'].fillna("No Dependents Data",inplace=True)

train_df['Self_Employed'].fillna("No Self_Employed Data",inplace=True)

train_df['Loan_Amount_Term'].fillna("No Loan_Amount_Term Data",inplace=True)

train_df['Credit_History'].fillna("No Credit_History Data",inplace=True)

train_df['LoanAmount'].fillna("No LoanAmount Data",inplace=True)
train_df.isnull().sum()
train_df.drop("Loan_ID", axis=1, inplace=True)
train_df.duplicated().any()
train_df.describe(include='O')
train_df.shape
train_df.select_dtypes(exclude=np.number).columns
"""for col in df.columns:

    print(col,':', df[df[col] == '?'][col].count())

    

for cols in df.select_dtypes(exclude=np.number).columns:

    df[cols] = df[cols].str.replace('?', 'Unknown')

    

for cols in df.select_dtypes(exclude=np.number).columns:

    print(cols, ':', df[cols].unique(), end='\n\n')

"""
# Encoding categorical features



categorical_columns = train_df.select_dtypes(exclude=np.number).columns

train_df2 = pd.get_dummies(data=train_df, prefix=categorical_columns, drop_first=True)
train_df2.shape
pd.set_option('max_columns', 232)

train_df2.head()
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from catboost import CatBoostClassifier

from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler



X=train_df2.drop('Loan_Status_Y',axis=1) #axis=1 drop the bulk of column

y=train_df2['Loan_Status_Y']



#X=train_df.drop('Loan_Status',axis=1) #axis=1 drop the bulk of column

#y=train_df['Loan_Status']



X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.25,random_state=0)



StSc = StandardScaler()

X_train  = StSc.fit_transform(X_train)

X_test  = StSc.fit_transform(X_test)
penalty = ['l2', 'elasticnet']

#C = np.logspace(-4,4,20)

C =[0.001,0.01,0.1,1,10,100]

solver=['lbfgs','liblinear']

l1_ratio=[0.001, 0.01, 0.1]



hyperparameters = dict(penalty=penalty, C=C, solver=solver, l1_ratio=l1_ratio)



logreg = LogisticRegression()



cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)



clf = GridSearchCV(logreg, hyperparameters, cv=cv, verbose=0)



best_model=clf.fit(X_test,y_test)
best_model.best_params_
y_pred=best_model.predict(X_test)
from sklearn.metrics import classification_report, roc_auc_score



print(classification_report(y_test,y_pred))

roc_auc_score(y_test,y_pred)
criterion = ['gini','entropy']

max_depth=[5,6,7,8,9]

n_estimator=[50,100,200,300,400,500]



hyperparameters = dict(criterion=criterion, max_depth=max_depth, n_estimators=n_estimator)



randomforest = RandomForestClassifier()



cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)



clf2 = GridSearchCV(randomforest, param_grid=hyperparameters, cv=cv, verbose=0)



#estimator.get_params().keys()
best_model2=clf2.fit(X_train, y_train)
best_model2.best_params_
y_pred2=best_model2.predict(X_test)



print(classification_report(y_test,y_pred2))

roc_auc_score(y_test,y_pred2)
max_depth=[2,3,4,5,6,7,8,9,10]

learning_rate=[0.001,0.01,0.1,0.2,0.3]

min_child_weight=[2,3,4,5,6,7]



hyperparameters = dict(max_depth=max_depth, learning_rate=learning_rate, min_child_weight=min_child_weight)



xgb = XGBClassifier()



cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)



clf3 = GridSearchCV(xgb, param_grid=hyperparameters, cv=cv, verbose=0)



#best_model3=clf3.fit(X_train, y_train)

clf3.fit(X_train, y_train)
import matplotlib.pyplot as plt



print("Feature importance by XGBoost:->\n")

XGBR = XGBClassifier()

XGBR.fit(X,y)

features = XGBR.feature_importances_

Columns = list(X.columns)

for i,j in enumerate(features):

    print(Columns[i],"->",j)

plt.figure(figsize=(16,5))

plt.title(label="XGBC")

plt.bar([x for x in range(len(features))],features)

plt.show()
y_pred3=clf3.predict(X_test)



print(classification_report(y_test,y_pred3))

roc_auc_score(y_test,y_pred3)
final_predictions = clf3.predict_proba(X_test)



print(roc_auc_score(y_test, final_predictions[:, 1]))