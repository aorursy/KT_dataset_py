import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score

import xgboost
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
df = pd.read_csv('../input/churn-modelling/Churn_Modelling.csv')
df.head()
df.shape
df.dtypes
df.isnull().sum()
df[df.columns[2:]].corr()['Exited'][:]
co = df[df.columns[2:]].corr()['Exited'][:]
features = co.index

plt.figure(figsize = (10, 5))
sns.heatmap(df[features].corr(), annot = True, cmap = 'viridis')
plt.show()
X = df.iloc[:, 3:13]
y = df.iloc[:, -1]
X = pd.get_dummies(X, columns = ['Geography', 'Gender'], drop_first = True)
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
lr = LogisticRegression()

# Train the model
model = lr.fit(X_train, y_train)

# Prediction
y_pred_test = lr.predict(X_test)
y_pred_train = lr.predict(X_train)

# Accuracy Score
print('Train Accuracy score : {}\n'.format(accuracy_score(y_train, y_pred_train)))
print('Test Accuracy score : {}'.format(accuracy_score(y_test, y_pred_test)))
cl1 = XGBClassifier()

# Train the model
model1 = cl1.fit(X_train, y_train)

# Prediction
y1_train_pred = cl1.predict(X_train)
y1_test_pred = cl1.predict(X_test)

# Accuracy Score
print('Train Accuracy score : {}\n'.format(accuracy_score(y_train, y1_train_pred)))
print('Test Accuracy score : {}'.format(accuracy_score(y_test, y1_test_pred)))
params = {
    'learning_rate'     : [0.05, 0.10, 0.05, 0.20, 0.25, 0.30],
    'max_depth'         : [3, 4, 5, 6, 8, 10, 12, 15],
    'min_child_weight'  : [1, 3, 5, 7],
    'gamma'             : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'colsmaple_bytree'  : [0.3, 0.4, 0.5, 0.6, 0.7]
    }
cl2 = XGBClassifier()
random_search = RandomizedSearchCV(cl2, param_distributions = params,
                                   n_iter = 5, scoring = 'roc_auc',
                                   n_jobs = -1,
                                   cv = 5,
                                   verbose = 3)
random_search.fit(X_train, y_train)
random_search.best_estimator_
random_search.best_params_
# Prediction
y2_train_pred = random_search.predict(X_train)
y2_test_pred = random_search.predict(X_test)

# Accuracy Score
print('Train Accuracy score : {}\n'.format(accuracy_score(y_train, y2_train_pred)))
print('Test Accuracy score : {}'.format(accuracy_score(y_test, y2_test_pred)))