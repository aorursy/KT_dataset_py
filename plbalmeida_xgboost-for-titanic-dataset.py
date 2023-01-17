# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
# load train and test datasets
train = pd.read_csv("../input/titanic/train.csv")
test = pd.read_csv("../input/titanic/test.csv")
submission = pd.read_csv("../input/titanic/gender_submission.csv")
# exploring first observations of train dataset
print(train.shape)
train.head()
# feature type
train.dtypes
# checking missing values
print(train.isnull().sum())
# simple preprocessing
def preprocessing(df):
    
    df['CabinBool'] = df['Cabin'].notnull().astype('int') # create boolean for "Cabin"
    df['Pclass'] = df['Pclass'].astype('str') # converting to object
    df = df.drop(['PassengerId','Name','Ticket', 'Cabin'], axis=1) # droping some features
    df = pd.get_dummies(df) # get dummies
    
    return df
train = preprocessing(train)
# exploring first observations of train dataset after preprocessing
train.head()
train = train.dropna()
# counting survivors and checking data balance
train.groupby('Survived')['Survived'].count()
# separate the target feature and rest of the features
X = train.drop('Survived', axis=1)
y = train.Survived

# train and test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123, stratify = y)
'''XGboost model with grid search'''

# parameter grid
gbm_param_grid = {
   'learning_rate': [0.05, 0.1, 0.15],
   'colsample_bytree':[0.7, 0.8, 0.9],
   'n_estimators':[50 ,75, 100],
   'max_depth':[11, 12, 13]
}

# instantiate the classifier
gbm = xgb.XGBClassifier(random_state=123)

# perform grid search
grid = GridSearchCV(
    estimator=gbm, 
    param_grid=gbm_param_grid,
    scoring='accuracy', 
    cv=10, 
    verbose=0
)

grid.fit(X_train, np.ravel(y_train))
print('Best parameters found: ', grid.best_params_)
print('Best accuracy found (train): ', grid.best_score_)
# preprocessing target test
test_pp = preprocessing(test)
test_pp[:5]
print(classification_report(submission.Survived, grid.predict(test_pp)))
pd.DataFrame({'PassengerId':submission.PassengerId,'Survived':grid.predict(test_pp)}).to_csv('submission.csv',index=False)