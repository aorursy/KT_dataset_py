import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv('../input/diabetes.csv')

df.head()
from sklearn.feature_selection import f_classif, chi2, mutual_info_classif

from statsmodels.stats.multicomp import pairwise_tukeyhsd





f_score, f_p_value = f_classif(X,y)

mut_info_score = mutual_info_classif(X,y)

# print('chi2 score        ', chi2_score)

# print('chi2 p-value      ', chi_2_p_value)

print('F - score score   ', f_score)

print('F - score p-value ', f_p_value)

print('mutual info       ', mut_info_score)
df.columns
X = df.drop(['Outcome'], axis=1)

X = StandardScaler().fit_transform(X)

y = df['Outcome']
from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE().fit_resample(X,y)
X_train, X_test, y_train, y_test = train_test_split(

    X_resampled, y_resampled, test_size=0.25, random_state=0)
model = SVC()



parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
grid = GridSearchCV(estimator=model, param_grid=parameters, cv=5)

grid.fit(X_train, y_train)
roc_auc = np.around(np.mean(cross_val_score(grid, X_test, y_test, cv=5, scoring='roc_auc')), decimals=4)

print('Score: {}'.format(roc_auc))
model = RandomForestClassifier(n_estimators=1000)

model.fit(X_train, y_train)

predictions = cross_val_predict(model, X_test, y_test, cv=5)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
score = np.mean(cross_val_score(model, X_test, y_test, cv=5, scoring='roc_auc'))

np.around(score, decimals=4)