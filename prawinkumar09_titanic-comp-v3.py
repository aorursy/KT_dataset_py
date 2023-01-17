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
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
train=pd.read_csv('/kaggle/input/titanic/train.csv')
test=pd.read_csv('/kaggle/input/titanic/test.csv')
train.head()
train.shape
train.describe()
train.isnull().sum()
train['Age'].fillna(28,inplace=True)
train.isnull().sum()
train['Embarked'].value_counts()
train['Embarked'].fillna('S',inplace=True)
train.isnull().sum()
train['Cabin'].value_counts()
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train.head()
train.drop(['Ticket','Cabin','Name'],axis=1,inplace=True)
train.head()

features = ['Sex','Pclass','SibSp','Embarked']

for f in features: 
    df = train[[f]]

    df1 = (pd.get_dummies(df, prefix='', prefix_sep='').max(level=0, axis=1).add_prefix(f+'_'))  
   
    train = pd.concat([train, df1], axis=1)

   
    train= train.drop([f], axis=1)
type(train)
plt.figure(figsize=(12,10))
cor = train.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)
plt.show()
cor_target = abs(cor["Survived"])
#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.2]
relevant_features
y = train.Survived
X = train.drop(['Survived'], axis=1).select_dtypes(exclude=['object'])
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3,random_state=5,stratify=y)

my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
from xgboost import XGBClassifier

my_model = XGBClassifier()

my_model.fit(train_X, train_y, verbose=False)
my_model.score(test_X, test_y)
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_validate,GridSearchCV
from sklearn import metrics 
import matplotlib.pylab as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
target='Survived'
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score

# Create the parameter grid: gbm_param_grid 
gbm_param_grid = {
    'n_estimators': range(700,1500),
    'max_depth': range(3, 10),
    'learning_rate': [.001, .05, .01, .015, .02],
    'colsample_bytree': [.6, .7, .8, .9, 1]
}

# Instantiate the regressor: gbm
gbm = XGBClassifier(n_estimators=10)

# Perform random search: grid_mse
xgb_random = RandomizedSearchCV(param_distributions=gbm_param_grid, 
                                    estimator = gbm, scoring = "accuracy", 
                                    verbose = 1, n_iter = 50, cv = 4)


# Fit randomized_mse to the data
xgb_random.fit(train_X,train_y)

# Print the best parameters and lowest RMSE
print("Best parameters found: ", xgb_random.best_params_)
print("Best accuracy found: ", xgb_random.best_score_)
predictions = xgb_random.predict(test_X)
print('Accuracy:',accuracy_score(test_y, predictions)*100)
alg = XGBClassifier(learning_rate=0.001, n_estimators=1022, max_depth=9,
                        min_child_weight=1, gamma=0.0, subsample=0.6, colsample_bytree=0.6,
                        objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
xgb_param = alg.get_xgb_params()
xgtrain = xgb.DMatrix(train_X, label=train_y)
cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=5,
                          early_stopping_rounds=10)
alg.set_params(n_estimators=cvresult.shape[0])
alg.fit(train_X, train_y, eval_metric='auc')
predictions1 = alg.predict(test_X)

print('Accuracy:',accuracy_score(test_y, predictions1)*100)
submission = pd.concat([test.PassengerId, pd.DataFrame(predictions)], axis = 'columns')
submission.columns = ["PassengerId", "Survived"]
submission.to_csv('titanic_submission.csv', header = True, index = False)