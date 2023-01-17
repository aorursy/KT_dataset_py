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

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('whitegrid')
df = pd.read_csv('/kaggle/input/bank-marketing-dataset/bank.csv')
df['loan'].value_counts()
df.head()
df.describe()
df.info()
plt.figure(figsize = (17, 6))

sns.countplot('age', hue = 'deposit', data = df)
plt.figure(figsize = (17, 6))

sns.countplot('education', hue = 'deposit', data = df)
plt.figure(figsize = (17, 6))

sns.countplot('job', hue = 'deposit', data = df)
plt.figure(figsize = (17, 6))

sns.countplot('housing', hue = 'deposit', data = df)
df.head()
plt.figure(figsize = (17, 6))

sns.countplot('loan', hue = 'deposit', data = df)
plt.figure(figsize = (17, 6))

sns.countplot('month', hue = 'deposit', data = df)
plt.figure(figsize = (17, 6))

sns.countplot('day', hue = 'deposit', data = df)
def impute(col):

    if col <= 4:

        return 1

    if col > 4 and col <= 10:

        return 2

    if col > 10 and col <= 13:

        return 3

    if col > 14 and col <= 21:

        return 4

    if col > 21:

        return 5
df['day_bool'] = df['day'].apply(impute)

plt.figure(figsize = (17, 6))

sns.countplot('day_bool', hue = 'deposit', data = df)
plt.figure(figsize = (17, 6))

sns.countplot('campaign', hue = 'deposit', data = df)
avg_duration = df['duration'].mean()
avg_duration
def impute(col):

    if col < avg_duration:

        return 'below_average'

    if col > avg_duration:

        return 'above_average'
df['duration_bool'] = df['duration'].apply(impute)
plt.figure(figsize = (17, 6))

sns.countplot('duration_bool', hue = 'deposit', data = df)
plt.figure(figsize = (17, 6))

sns.countplot('previous', hue = 'deposit', data = df)
plt.figure(figsize = (17, 6))

sns.countplot('poutcome', hue = 'deposit', data = df)
df.drop(['day_bool', 'duration_bool', 'pdays'], axis = 1, inplace = True)
df['deposit']=df['deposit'].map({'yes':1,'no':0})

df = pd.get_dummies(df, columns=['job','marital','education',"month",'default','housing',"loan","contact","poutcome"], drop_first=True)
df.head()
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV



from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb



from sklearn.metrics import confusion_matrix, classification_report
X = df.drop('deposit', axis = 1)

y = df['deposit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier())

gs_knn = GridSearchCV(estimator=pipeline, param_grid={'kneighborsclassifier__n_neighbors': [3,4,5,6,7]}, scoring='accuracy', cv = 10)

knn_score = cross_val_score(gs_knn, X = X_train, y = y_train, cv = 5, scoring='accuracy', n_jobs=-1)

gs_knn.fit(X_train, y_train)

gs_best = gs_knn.best_estimator_

gs_best.fit(X_train, y_train)
prediction=gs_best.predict(X_test)

print(confusion_matrix(prediction ,y_test))

print(classification_report(prediction ,y_test))
pipeline = make_pipeline(StandardScaler(), LogisticRegression())

gs_lr = GridSearchCV(estimator=pipeline, param_grid={'logisticregression__C': np.arange(1, 5), 'logisticregression__max_iter': [100, 300, 1000, 3000]}, scoring = 'accuracy', cv = 10)

gs_score = cross_val_score(gs_lr, X = X_train, y = y_train, cv = 5, scoring='accuracy', n_jobs=-1)

gs_lr.fit(X_train, y_train)

gs_best = gs_lr.best_estimator_

gs_best.fit(X_train, y_train)

prediction=gs_best.predict(X_test)

print(confusion_matrix(prediction ,y_test))

print(classification_report(prediction ,y_test))
rf = RandomForestClassifier()

gs_rf = GridSearchCV(estimator = rf, param_grid={'n_estimators': [100, 300, 400]}, scoring='accuracy', cv = 2)

gs_score = cross_val_score(gs_rf, X = X_train, y = y_train, cv = 5, scoring='accuracy', n_jobs=-1)

gs_rf.fit(X_train, y_train)

gs_best = gs_rf.best_estimator_

gs_best.fit(X_train, y_train)

prediction=gs_best.predict(X_test)

print(confusion_matrix(prediction ,y_test))

print(classification_report(prediction ,y_test))
feature = gs_best.feature_importances_

feature_importances = pd.Series(feature, index=X_train.columns).sort_values(ascending = False)

sns.barplot(x=feature_importances[0:10], y=feature_importances.index[0:10])

sns.despine()

plt.xlabel("Feature Importances")

plt.ylabel("Features")