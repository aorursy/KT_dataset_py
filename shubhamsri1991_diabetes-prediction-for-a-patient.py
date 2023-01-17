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
diabetes_dataset = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')
diabetes_dataset.head()
diabetes_dataset.info()
diabetes_dataset.describe()
diabetes_dataset.isnull().sum()
unique = diabetes_dataset.nunique()
unique
diabetes_dataset.duplicated().sum()
import matplotlib.pyplot as plt
import seaborn as sns
corr = diabetes_dataset.corr()
plt.figure(figsize=(12,5))
sns.heatmap(corr, annot=True,cmap = "coolwarm")
plt.show()
from sklearn.feature_selection import f_classif
label = diabetes_dataset['Outcome']
fval, pval = f_classif(diabetes_dataset, label)
for i in range(len(pval)):
    print(diabetes_dataset.columns[i], pval[i].round(4))
from sklearn.feature_selection import chi2
fval, pval = chi2(diabetes_dataset, label)
for i in range(len(pval)):
    print(diabetes_dataset.columns[i], pval[i].round(4))
import seaborn as sns
sns.pairplot(diabetes_dataset, hue = 'Outcome')
features = diabetes_dataset[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
features.head()
label.head()
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=20, random_state=None, shuffle=False)
skf.get_n_splits(features, label)
for train_index, test_index in skf.split(features, label):
    features_train, features_test = features.iloc[train_index], features.iloc[test_index]
    label_train, label_test = label.iloc[train_index], label.iloc[test_index]
features_train.shape
features_test.shape
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
randam_forest_model.fit(features_train, label_train)
label_pred = randam_forest_model.predict(features_test)
label_pred
from sklearn.metrics import recall_score
recall_score(label_test, randam_forest_model.predict(features_test))
from sklearn.metrics import recall_score
recall_score(label_train, randam_forest_model.predict(features_train))
from sklearn.metrics import precision_score
precision_score(label_test, randam_forest_model.predict(features_test))
from sklearn.metrics import precision_score
precision_score(label_train, randam_forest_model.predict(features_train))
from sklearn.metrics import accuracy_score
accuracy_score(label_test, randam_forest_model.predict(features_test))
from sklearn.metrics import accuracy_score
accuracy_score(label_train, randam_forest_model.predict(features_train))