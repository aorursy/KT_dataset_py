import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
train = pd.read_csv('../input/train_data.csv',
        names=[
        "Id", "Age", "Workclass", "Final Weight", "Education", "Education-Num", 
        "Martial Status", "Occupation", "Relationship", "Race", "Sex", "Capital Gain", 
        "Capital Loss", "Hours per week", "Country", "Target"],
        index_col=0,
        skiprows=1,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
train.head()
test = pd.read_csv('../input/test_data.csv',
        index_col=0,
        names=[
        "Id", "Age", "Workclass", "Final Weight", "Education", "Education-Num", 
        "Martial Status", "Occupation", "Relationship", "Race", "Sex", "Capital Gain", 
        "Capital Loss", "Hours per week", "Country"],
        skiprows=1,
        sep=r'\s*,\s*',
        engine='python',
        na_values="?")
test.head()
ntrain = train.dropna()
ntrain['Target'] = ntrain['Target'].apply(lambda x: 0 if x == '<=50K' or x == '<=50K.' else 1)
x_train = ntrain.drop('Target', axis=1)
y_train = ntrain['Target']
ntest = test.dropna()
x_train.shape
import seaborn as sns
sns.pairplot(ntrain, hue='Target')
plt.show()
sns.barplot(x='Country', y='Target', data=ntrain)
plt.xticks(rotation=90)
plt.show()
sns.barplot(x='Age', y='Target', data=ntrain)
sns.set_context('notebook', font_scale=0.5)
plt.xticks(rotation=90)
plt.show()
sns.barplot(x='Hours per week', y='Target', data=ntrain)
sns.set_context('notebook', font_scale=0.5)
plt.xticks(rotation=90)
plt.show()
numerical_features = ['Age', 'Final Weight', 'Education-Num', 'Capital Gain', 'Capital Loss', 'Hours per week']
categorical_features = ['Workclass', 'Education', 'Martial Status', 'Occupation', 'Relationship', 'Race', 'Sex', 'Country']
final_x_train = train.dropna().drop('Target', axis=1)
final_test = test.fillna(method='ffill')
from sklearn.preprocessing import OneHotEncoder
categorical_indices = [final_x_train.columns.get_loc(col) for col in categorical_features]
print(categorical_indices)
encoder = OneHotEncoder(categorical_features=categorical_indices)
#one_hot_x_train = encoder.fit_transform(final_x_train)
#one_hot_x_train.head()
from scipy.stats import randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
def create_csv(pred, test, name):
    results = np.vstack((test.index, pred)).T
    cols = ['Id', 'income']
    df_pred = pd.DataFrame(columns=cols ,data=results)
    df_pred['Id'] = df_pred['Id'].astype('Int32') 
    df_pred.to_csv(name, index=False)
from sklearn.neighbors import KNeighborsClassifier
knn_params = {'n_neighbors': randint(1, 31)}
knn_clf = KNeighborsClassifier()
knn_search = RandomizedSearchCV(knn_clf, param_distributions=knn_params,
                               n_iter=15, cv=10, scoring='accuracy', return_train_score=False)
best_knn = knn_search.fit(final_x_train[numerical_features], y_train)
print('Melhor K: {}'.format(best_knn.best_estimator_.get_params()['n_neighbors']))
knn_pred = best_knn.predict(final_test[numerical_features])
create_csv(knn_pred, final_test, 'knn_pred.csv')
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=0, n_jobs=-1)
rf_params = {'max_depth': [1, 2, 3, None],
             'n_estimators': [20, 50, 100, 200],
             'criterion': ['gini', 'entropy'],
             'max_features': ['auto', 'sqrt', 'log2']}
from time import time
inicio = time()
rf_search = GridSearchCV(rf_clf, param_grid=rf_params,
                        cv=10, scoring='accuracy', return_train_score=False)
best_rf = rf_search.fit(final_x_train[numerical_features], y_train)
print('Demorou {} segundos'.format(time()-inicio))
print('Melhores parâmetros: {}'.format(best_rf.best_estimator_.get_params()))
rf_pred = best_rf.predict(final_test[numerical_features])
create_csv(rf_pred, final_test, 'rf_pred.csv')
from xgboost import XGBClassifier
xgb_clf = XGBClassifier()
xgb_params = {'max_depth': list(range(1,6)),
             'booster': ['gbtree', 'gblinear', 'dart']}
inicio = time()
xgb_search = RandomizedSearchCV(xgb_clf, param_distributions=xgb_params,
                               n_iter=15, cv=10, scoring='accuracy', return_train_score=False)
best_xgb = knn_search.fit(final_x_train[numerical_features], y_train)
print('Demorou {} segundos'.format(time()-inicio))
print('Melhores parâmetros: {}'.format(best_xgb.best_estimator_.get_params()))
xgb_pred = best_xgb.predict(final_test[numerical_features])
create_csv(xgb_pred, final_test, 'xgb_pred.csv')
