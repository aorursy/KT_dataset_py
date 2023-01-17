import os
from scipy import stats
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import RandomizedSearchCV,train_test_split,StratifiedKFold
import xgboost as xgb
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
submission = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
print(f"Train Data Shape: {train.shape}")
print(f"Test Data Shape: {test.shape}")
submission.head()
train.head(5)
train.dtypes
## Missing values
train.isnull().sum()*100/train.shape[0]
## Unique Values of Cabin
len(train.Cabin.unique())
train.Sex.value_counts()
train.Pclass.value_counts()
train.SibSp.value_counts()
train.Parch.value_counts()
train.Embarked.value_counts()
onehot_cols = cat_cols = ['Embarked','Parch','SibSp','Pclass','Sex']
train = pd.get_dummies(data = train, columns = onehot_cols)
train = train.drop(['PassengerId','Ticket','Name','Cabin','Sex_male'],axis = 1)
train.head()
test = pd.get_dummies(data = test, columns = onehot_cols)
test = test.drop(['Ticket','Name','Cabin','Sex_male'],axis = 1)
test.head()
## Train test Split
y = train['Survived']
X = train.drop('Survived',axis = 1)
## Model
# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    'n_estimators': range(10, 50),
    'max_depth': range(5, 10),
    'learning_rate': [0.01, 0.05, 0.1, 0.25, 0.50],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 2, 3, 4]
}

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 42)

# Instantiate the regressor: gbm
gbm = xgb.XGBClassifier(objective = 'binary:logistic')

# Perform random search: grid_mse
xgb_random = RandomizedSearchCV(gbm, 
                         param_distributions = gbm_param_grid,
                         cv = skf.split(X,y),  
                         n_iter = 5,
                         scoring = 'roc_auc',  
                         verbose = 3, 
                         n_jobs = -1)


# Fit randomized_mse to the data
xgb_random.fit(X, y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", xgb_random.best_params_)
print("Best accuracy found: ", xgb_random.best_score_)
ids = test.PassengerId
y_test = xgb_random.predict(test[X.columns])
submission = pd.DataFrame({'PassengerId':ids,'Survived':y_test})
submission.head()
submission.Survived.value_counts()
submission.to_csv('submision_xgb.csv',index=False)
