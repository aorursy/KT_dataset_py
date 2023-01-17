from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import Imputer, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

import pandas as pd

import numpy as np
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')
df_train.head()
df_train.info()
df_train.isna().sum()/df_train.shape[0]
y=df_train.Survived

X=df_train.drop(['PassengerId','Survived'], axis=1)



X_train_full, X_valid_full,y_train,y_valid = train_test_split(X,y, test_size=0.2,random_state=0)





# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns 

                  if X_train_full[cname].dtype in ['int64', 'float64']]





# Select categorical columns

categorical_cols = [cname for cname in X_train_full.columns 

                    if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]





# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = df_test[my_cols].copy()
X_train.head()
# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy ='median')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



#Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators=100, random_state = 0, class_weight='balanced')



model_xgb = XGBClassifier(n_estimators = 50, learning_rate=0.01, n_jobs=4)
# Bundle preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor',preprocessor),

                             ('model', model_xgb)])



my_pipeline.fit(X_train,y_train)



# Prediction

preds = my_pipeline.predict(X_valid)



print (accuracy_score(y_valid, preds))
scores = cross_val_score(my_pipeline, X,y, cv=5,scoring='accuracy')



#Random Forest: 0.803

#XGB Classifier:0.818

print ("Average accuracy scores:\n", scores.mean())
preds_test = my_pipeline.predict(X_test)





output= pd.DataFrame({'PassengerId': df_test.PassengerId,

                     'Survived': preds_test})



output.to_csv('titanic.csv', index=False)