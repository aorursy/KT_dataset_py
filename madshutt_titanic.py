import pandas as pd
import numpy as np

%matplotlib inline
from scipy.stats import norm
import scipy.stats as st
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.preprocessing.imputation import Imputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

data.head()
pd.get_dummies(data, columns=['Sex']).corr()
# Correlation Matrix (heatmap)
corrmat = pd.get_dummies(data, columns=['Sex']).corr()
f, ax = plt.subplots(figsize=(16, 9))
sns.heatmap(corrmat, vmax=.8, square=True, annot=True, cmap="RdBu_r") # altri valori: BuGn_r, BrBG
# SNS Graph: Men and women count
sns.countplot(data['Sex'])
# data = pd.get_dummies(titanic_data, columns=['Sex'])
# Age of people
plt.title("Age (Survived in orange)")
data['Age'].plot.hist(edgecolor='black', linewidth=0.5)
data[data.Survived == 1]['Age'].plot.hist(edgecolor='black', linewidth=0.5)
plt.xlabel("Age")
plt.ylabel("Persons")
# Men's age
data[(data.Sex == 'male')]['Age'].plot.hist(edgecolor='black', linewidth=0.5)
data[(data.Survived == 1) & (data.Sex == 'male')]['Age'].plot.hist(edgecolor='black', linewidth=0.5)
# Women's age
data[(data.Sex == 'female')]['Age'].plot.hist(edgecolor='black', linewidth=0.5)
data[(data.Survived == 1) & (data.Sex == 'female')]['Age'].plot.hist(edgecolor='black', linewidth=0.5)
#titanic_data.head()
#test_data.head()

# List of features, will be used on test_data too

features = ['Pclass', 'Age', 'Fare']

X = data [features] #V27: features (0.8086)
y = data['Survived']
# Split into validation and training data
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.50, random_state=1)

# Parameters
n_est = 1000
learn = 0.10
max_dp = 3

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Modello: XGBClassifier
xgb_model = XGBClassifier (n_estimators=n_est, learning_rate=learn, max_depth=max_dp)
xgb_model.fit (train_X, train_y)
xgb_predictions = xgb_model.predict (test_X)

print("Accuracy: {0}".format(accuracy_score(test_y, xgb_predictions)))
# XGBRegressor Pipeline with full data and Cross-Validation #V27 = 0.8060
xgb_final = make_pipeline(XGBClassifier(n_estimators=n_est, 
                                        learning_rate=learn, 
                                        xgbclassifier__early_stopping_rounds=5, 
                                        xgbclassifier__eval_set=[(X, y)]))

# Cross-Validation
scores = cross_val_score(xgb_final, X, y, scoring='accuracy', cv=3)
print('XGB Pipeline Cross-Validation Accuracy: %2f' %scores.mean())
print(scores)

xgb_final.fit(X, y);
ids = test['PassengerId']
final_data = test [features]

predictions = xgb_final.predict(final_data);

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)
output.head()


