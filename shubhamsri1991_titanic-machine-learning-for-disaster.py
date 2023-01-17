# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
titanic_train_data = pd.read_csv('../input/titanic/train.csv')
titanic_test_data = pd.read_csv('../input/titanic/test.csv')
pd.pandas.set_option('display.max_columns', None)
titanic_train_data.head()
pd.pandas.set_option('display.max_columns', None)
titanic_test_data.head()
titanic_train_data.shape, titanic_test_data.shape
titanic_train_data.info(), titanic_test_data.info()
titanic_train_data.isnull().sum().plot(kind='bar')
titanic_test_data.isnull().sum().plot(kind='bar')
sns.heatmap(titanic_train_data.isnull(), cbar=False)
sns.heatmap(titanic_test_data.isnull(), cbar=False)
titanic_train_data.drop(['Cabin'], axis=1, inplace=True)
titanic_test_data.drop(['Cabin'], axis=1, inplace=True)
titanic_train_data.describe()
titanic_test_data.describe()
sns.boxplot(titanic_train_data['Age'])
sns.boxplot(titanic_train_data['SibSp'])
sns.boxplot(titanic_train_data['Parch'])
sns.boxplot(titanic_train_data['Fare'])
plt.hist(titanic_train_data['Age'], bins=50)
plt.show()
plt.hist(titanic_train_data['SibSp'], bins=50)
plt.show()
plt.hist(titanic_train_data['Parch'], bins=50)
plt.show()
plt.hist(titanic_train_data['Fare'], bins=50)
plt.show()
titanic_train_data.duplicated().sum(), titanic_test_data.duplicated().sum()
titanic_train_data['Age'].fillna(titanic_train_data['Age'].median(), inplace=True)
titanic_train_data['Embarked'].fillna('Missing', inplace=True)
titanic_test_data['Age'].fillna(titanic_test_data['Age'].mean(), inplace=True)
titanic_test_data['Fare'].fillna(titanic_test_data['Fare'].median(), inplace=True)
titanic_train_data.info(), titanic_test_data.info()
train_dataset_outliers_features = ['Age', 'SibSp', 'Parch', 'Fare']
for feature in train_dataset_outliers_features:
    IQR = titanic_train_data[feature].quantile(0.75) - titanic_train_data[feature].quantile(0.25)
    train_dataset_outliers_feature_lower_boundary = titanic_train_data[feature].quantile(0.25) - (3*IQR)
    train_dataset_outliers_feature_upeer_boundary = titanic_train_data[feature].quantile(0.75) + (3*IQR)
    print(feature, train_dataset_outliers_feature_lower_boundary, train_dataset_outliers_feature_upeer_boundary)
for feature in train_dataset_outliers_features:
    IQR = titanic_train_data[feature].quantile(0.75) - titanic_train_data[feature].quantile(0.25)
    train_dataset_outliers_feature_lower_boundary = titanic_train_data[feature].quantile(0.25) - (3*IQR)
    train_dataset_outliers_feature_upeer_boundary = titanic_train_data[feature].quantile(0.75) + (3*IQR)
    titanic_train_data.loc[titanic_train_data[feature] <= train_dataset_outliers_feature_lower_boundary, feature] = train_dataset_outliers_feature_lower_boundary
    titanic_train_data.loc[titanic_train_data[feature] >= train_dataset_outliers_feature_upeer_boundary, feature] = train_dataset_outliers_feature_upeer_boundary
outliers_features = ['Age', 'SibSp', 'Parch', 'Fare']
for feature in outliers_features:
    IQR = titanic_test_data[feature].quantile(0.75) - titanic_test_data[feature].quantile(0.25)
    lower_boundary = titanic_test_data[feature].quantile(0.25) - (3*IQR)
    upeer_boundary = titanic_test_data[feature].quantile(0.75) + (3*IQR)
    print(feature, lower_boundary, upeer_boundary)
for feature in outliers_features:
    IQR = titanic_test_data[feature].quantile(0.75) - titanic_test_data[feature].quantile(0.25)
    lower_boundary = titanic_test_data[feature].quantile(0.25) - (3*IQR)
    upper_boundary = titanic_test_data[feature].quantile(0.75) + (3*IQR)
    titanic_test_data.loc[titanic_test_data[feature] <= lower_boundary, feature] = lower_boundary
    titanic_test_data.loc[titanic_test_data[feature] >= upper_boundary, feature] = upper_boundary
titanic_test_data_id = titanic_test_data['PassengerId']
plt.figure(figsize=(12,5))
titanic_train_data_corr = titanic_train_data.corr()
sns.heatmap(titanic_train_data_corr, annot=True, cmap='RdYlGn')
plt.show()
titanic_train_data.drop(['PassengerId', 'Parch'], axis=1, inplace=True)
titanic_test_data.drop(['PassengerId', 'Parch'], axis=1, inplace=True)
titanic_train_data['Sex'] = pd.get_dummies(titanic_train_data['Sex'], drop_first=True)
titanic_train_data['Name'] = pd.get_dummies(titanic_train_data['Name'], drop_first=True)
titanic_train_data['Ticket'] = pd.get_dummies(titanic_train_data['Ticket'], drop_first=True)
titanic_train_data['Embarked'] = pd.get_dummies(titanic_train_data['Embarked'], drop_first=True)
titanic_test_data['Sex'] = pd.get_dummies(titanic_test_data['Sex'], drop_first=True)
titanic_test_data['Name'] = pd.get_dummies(titanic_test_data['Name'], drop_first=True)
titanic_test_data['Ticket'] = pd.get_dummies(titanic_test_data['Ticket'], drop_first=True)
titanic_test_data['Embarked'] = pd.get_dummies(titanic_test_data['Embarked'], drop_first=True)
titanic_train_data_features = titanic_train_data.drop(['Survived'], axis=1)
titanic_train_data_label = titanic_train_data['Survived']
plt.figure(figsize=(12,10))
from sklearn.ensemble import ExtraTreesRegressor
etr = ExtraTreesRegressor()
etr.fit(titanic_train_data_features, titanic_train_data_label)
feat_importances = pd.Series(etr.feature_importances_, index=titanic_train_data_features.columns)
feat_importances.nlargest(20).plot(kind='barh')
plt.show
titanic_train_data_features.drop(['Name', 'Embarked','Ticket'], axis=1, inplace=True)
titanic_test_data.drop(['Name','Embarked', 'Ticket'], axis=1, inplace=True)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=20, random_state=None, shuffle=False)
skf.get_n_splits(titanic_train_data_features, titanic_train_data_label)
for train_index, test_index in skf.split(titanic_train_data_features, titanic_train_data_label):
    train_features, test_features = titanic_train_data_features.iloc[train_index], titanic_train_data_features.iloc[test_index]
    train_label, test_label = titanic_train_data_label.iloc[train_index], titanic_train_data_label.iloc[test_index]
#Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]

#Number of features to consider in every split
max_features = ['auto', 'sqrt']

#Maximum number of levels in a tree
max_depth = [int(x) for x in np.linspace(start=5, stop=30, num=6)]

#Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]

#Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
#Random Grid
random_grid = {'n_estimators' : n_estimators,
              'max_features' : max_features,
              'max_depth' : max_depth,
              'min_samples_split' : min_samples_split,
              'min_samples_leaf' : min_samples_leaf}
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
random_forest = RandomForestClassifier()
randam_forest_model = RandomizedSearchCV(estimator=random_forest, param_distributions=random_grid,
                                         scoring='neg_mean_squared_error', n_iter=10, cv=5, verbose=2,
                                        random_state=42, n_jobs=1)
randam_forest_model.fit(train_features, train_label)
randam_forest_model.best_params_
from sklearn.metrics import accuracy_score
accuracy_score(train_label, randam_forest_model.predict(train_features))
from sklearn.metrics import accuracy_score
accuracy_score(test_label, randam_forest_model.predict(test_features))
from sklearn.metrics import recall_score
recall_score(train_label, randam_forest_model.predict(train_features))
from sklearn.metrics import recall_score
recall_score(test_label, randam_forest_model.predict(test_features))
from sklearn.metrics import precision_score
precision_score(train_label, randam_forest_model.predict(train_features))
from sklearn.metrics import precision_score
precision_score(test_label, randam_forest_model.predict(test_features))
titanic_test_dataset_label = randam_forest_model.predict(titanic_test_data)
titanic_test_dataset_label
test_dataset_label = pd.DataFrame(titanic_test_dataset_label)
titanic_data_prediction_submission = pd.concat([titanic_test_data_id, test_dataset_label], axis=1)
titanic_data_prediction_submission.columns = ['PassengerId', 'Survived']
titanic_data_prediction_submission.to_csv('titanic_data_prediction_submission.csv', index=False)