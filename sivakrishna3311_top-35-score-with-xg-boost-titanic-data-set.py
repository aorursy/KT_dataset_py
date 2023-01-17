import pandas as pd
import numpy as np
import warnings 
warnings.filterwarnings('ignore')
train_df=pd.read_csv('../input/titanic/train.csv')
train_df
test_df=pd.read_csv('../input/titanic/test.csv')
train_df.isnull().sum()
# Drop the Cabin column as it has close to 80% missing data. Will not make sense to Impute it.
train_df.drop('Cabin', axis=1, inplace=True)
test_df.drop('Cabin', axis=1, inplace=True)

train_df.isnull().sum()
#Missing value imputation
columns=['Age','Embarked']
for i in columns:
    train_df[i].fillna(value=np.nan,inplace=True)
#Replacing np.nan with average
train_df['Age'].fillna(train_df['Age'].mean(),inplace=True)
#Replacing np.nan with mode for Embarked
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0],inplace=True)
test_df.isnull().sum()
columns=['Age','Fare']
for i in columns:
    test_df[i].fillna(value=np.nan,inplace=True)
#Replacing np.nan with average
test_df['Age'].fillna(test_df['Age'].mean(),inplace=True)
test_df['Fare'].fillna(test_df['Fare'].mean(),inplace=True)
# Dropping irrelaveant features from test and train sets. Also alloting dependent var "y" and independent vars "X".

y = train_df['Survived'].astype(int)
X = train_df.drop(['PassengerId', 'Survived', 'Name', 'Ticket'], axis=1)
d2=test_df["PassengerId"]
n_test_df = test_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
cat_col=['object']
df_catcols_only=X.select_dtypes(include=cat_col)
df_catcols_only1=n_test_df.select_dtypes(include=cat_col)
df_catcols_only.columns
df_catcols_only1.columns
X=pd.get_dummies(data=X,columns=['Sex', 'Embarked'],drop_first=True)
n_test_df=pd.get_dummies(data=n_test_df,columns=['Sex', 'Embarked'],drop_first=True)
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=1)

print(Xtrain.shape)
print(Xtest.shape)
print(ytrain.shape)
print(ytest.shape)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

parameters = {"learning_rate": [0.1, 0.01, 0.001],
               "gamma" : [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2],
               "max_depth": [2, 4, 7, 10],
               "colsample_bytree": [0.3, 0.6, 0.8, 1.0],
               "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
               "reg_alpha": [0, 0.5, 1],
               "reg_lambda": [1, 1.5, 2, 3, 4.5],
               "min_child_weight": [1, 3, 5, 7],
               "n_estimators": [100, 250, 500, 1000]}

xgb_rscv = RandomizedSearchCV(XGBClassifier(), param_distributions = parameters, cv = 7,  random_state = 40)

model_rscv = xgb_rscv.fit(Xtrain, ytrain)
model_rscv.best_params_
# Best parameter values
tuned_model = XGBClassifier(booster='gbtree', subsample= 0.6,
 reg_lambda= 1,
 reg_alpha= 0,
 n_estimators= 100,
 min_child_weight= 5,
 max_depth= 2,
 learning_rate = 0.1,
 gamma= 1.5,
 colsample_bytree= 1.0)

tuned_model.fit(Xtrain, ytrain)
tuned_model.score(Xtest, ytest)
predictions = tuned_model.predict(n_test_df)

output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")