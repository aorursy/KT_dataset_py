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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#importing dataset
df = pd.read_csv('../input/sample-insurance-claim-prediction-dataset/insurance2.csv')
df.head()
#checking for null values
df.isnull().sum()
df.columns
X = df.iloc[:,:-1]
y = df['insuranceclaim']
X.head()
y.head()
df.head()
#Feature selection using EXtra Tree Classifier
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(X,y)
print(model.feature_importances_)
ranked_features = pd.Series(model.feature_importances_, index=X.columns)
ranked_features.nlargest(len(X.columns)).plot(kind='barh')
plt.show()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())
#Model Building
#Random Forest
from sklearn.ensemble import RandomForestClassifier
random_clf = RandomForestClassifier(n_estimators=900, min_samples_split=5, min_samples_leaf=5, max_features='sqrt', max_depth=10, criterion='entropy')
random_clf.fit(X_train,y_train)
random_clf_predict = random_clf.predict(X_test)
random_clf_predict
# #Performance Checking
from sklearn.metrics import confusion_matrix
random_clf_cm = confusion_matrix(y_test,random_clf_predict)
random_clf_cm
# #Classification Report
from sklearn.metrics import classification_report
random_clf_report = classification_report(y_test,random_clf_predict)
print(random_clf_report)
from sklearn.model_selection import RandomizedSearchCV
#Randomized Search CV

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]
criterion = ['entropy','gini']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num = 6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'criterion':criterion,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

print(random_grid)
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator = random_clf, param_distributions = random_grid,scoring='neg_mean_squared_error', n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)
rf_random.fit(X_train,y_train)
rf_random.best_params_
rf_random.best_score_
predictions=rf_random.predict(X_test)
predictions
print(classification_report(y_test, predictions))
import pickle
#open a file, where you want to store the data
file = open('rf_random_model.pkl','wb')

#dupming model to the file
pickle.dump(rf_random,file)