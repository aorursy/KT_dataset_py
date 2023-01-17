# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Lets input the data here

titanic_Train_data="../input/titanic/train.csv"

test_data='../input/titanic/test.csv'

test_df=pd.read_csv(test_data)

tita_train=pd.read_csv(titanic_Train_data)

tita_train=tita_train.drop('PassengerId',axis=1)

test_new=test_df.drop('PassengerId',axis=1)


from sklearn.model_selection import train_test_split

tita_train.dropna(axis=0,subset=['Survived'],inplace=True)

y=tita_train.Survived

X=tita_train.drop(["Survived"],axis=1)

X.drop('Name',axis=1)

test_new.drop('Name',axis=1)

X_initial=X.copy()
#Lets keep selected cat and num cols here

categorical_cols=[col for col in X_initial.columns if X_initial[col].nunique()<=15 and X_initial[col].dtype=='object']

numerical_cols=[coln for coln in X_initial.columns if X_initial[coln].dtype in ['int64','float64']]

my_cols=categorical_cols+numerical_cols

X_train=X_initial[my_cols].copy()


from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier 

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

numerical_transformer=SimpleImputer(strategy='constant')

categorical_transformer=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),

                                        ('onehot',OneHotEncoder(handle_unknown='ignore'))])

preprocessor=ColumnTransformer(transformers=[('nums',numerical_transformer,numerical_cols),

                                             ('cats',categorical_transformer,categorical_cols)])

model=XGBClassifier(random_state=0,n_estimators=200,learning_rate=0.1)





from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

X_final=preprocessor.fit_transform(X_train,y)

model.fit(X_final,y)

scores=cross_val_score(model,X_final,y,scoring='accuracy',cv=4)

print(scores.mean())
X_test=preprocessor.transform(test_new)

preds_test=model.predict(X_test)
submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": preds_test

    })

submission.to_csv('titanic_submission.csv', index=False)