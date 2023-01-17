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
# loading Dataframe ..
df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")
df.head()
# Check is any features have null values..
df.isnull().values.any()
# information about the data frame 
df.info()
# Numerical discription of the features..
df.describe()
## Importing Library  for correlation. 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
## to getting correlation of every feautres in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(15,15) )
# ploting heat map for visualisation ..
m = sns.heatmap(df[top_corr_features].corr(),annot= True, cmap="RdYlGn")
corrmat

df.head()
## checking the data for unbaised or not ..
diabetes_count = len(df.loc[df['Outcome'] == 1])
not_diabetes_count = len(df.loc[df['Outcome'] == 0])
(diabetes_count,not_diabetes_count)
## TRAIN TEST SPLIT

from sklearn.model_selection import train_test_split
x = df.iloc[:,0:-1]  #independent columns
y = df.iloc[:,-1]    #target column i.e outcome

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30,random_state = 42)
for feature in top_corr_features:
    print(f"Number of entries are missing are {len(df.loc[df[feature] == 0])}")
## Replacing zero values  to the mean of the feature
#from sklearn.preprocessing import Imputer

#fill_values = Imputer(missing_values = 0,strategy = 'mean',axis = 0)
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values= 0, strategy='mean')
x_test = imp.fit_transform(x_test)
x_train = imp.fit_transform(x_train)
## applying algorithm
from sklearn.ensemble import RandomForestClassifier
r_f_model = RandomForestClassifier(random_state=42)
r_f_model.fit(x_train,y_train.ravel())
## Predicting test values in the model that we have train .
predict_train_data = r_f_model.predict(x_test)

## checking the accuracy of my model
from sklearn import metrics

print(f"Accuracy = {metrics.accuracy_score(y_test, predict_train_data)},3f")

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)
## training


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x_train,y_train)
rf_random.best_estimator_
classifier = RandomForestClassifier(bootstrap=False, max_depth=90, min_samples_leaf=2,
                       min_samples_split=5, n_estimators=150)
from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,x,y.ravel(),cv=10)
score
score.mean()
