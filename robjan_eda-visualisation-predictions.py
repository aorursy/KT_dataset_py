import os
print("Data:\n",os.listdir("../input"))
import pandas as pd
import numpy as np
df = pd.read_csv('../input/adult.csv')
df_copy = df.copy()
df.head()
df.info()
df.describe()
[print(df[column].value_counts()) for column in df if df[column].dtype != np.int64]
sex = {'Male':0, 'Female':1}
df['sex'] = df['sex'].map(sex)
# 1 - USA, 0 - other
for i in range(len(df['native.country'])):
    country = df['native.country'].iloc[i]
    if country == 'United-States':
        df.at[i, 'native.country'] = 1
    else:
        df.at[i, 'native.country'] = 0
race = {'White':0, 'Black':1, 'Asian-Pac-Islander':2, 
        'Amer-Indian-Eskimo':3, 'Other':4}
df['race'] = df['race'].map(race)

income = {'<=50K': 0, '>50K': 1}
df['income'] = df['income'].map(income)
df.head()
# drop for now other columns
new_df = df.copy()
new_df.drop(['workclass', 'education', 'marital.status',
            'occupation', 'relationship'],axis=1, inplace=True)
new_df.head()
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(fig_id + '.png')
    print('Saving figure ', fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
corr_mat = new_df.corr()
sns.heatmap(corr_mat, xticklabels=corr_mat.columns, yticklabels=corr_mat.columns)
corr_mat['income'].sort_values(ascending=False)
new_df.drop('fnlwgt', axis=1, inplace=True)
from sklearn.model_selection import train_test_split
X = new_df.drop('income', axis=1)
y = new_df['income']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
from sklearn.linear_model import LogisticRegression

log_reg_ = LogisticRegression()
log_reg_.fit(x_train, y_train)
from sklearn.metrics import classification_report
y_log_pred = log_reg_.predict(x_test)
raport_log = classification_report(y_test, y_log_pred, labels=[0,1])
print(raport_log)
from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier()
dec_tree.fit(x_train, y_train)
y_tree_pred = dec_tree.predict(x_test)
raport_log = classification_report(y_test, y_tree_pred, labels=[0,1])
print(raport_log)
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth':[None,2,5,10,30,50],
             'min_samples_leaf':[2,0.2,0.5]}
tree_grid = GridSearchCV(dec_tree, param_grid, scoring='accuracy')
tree_grid.fit(x_train, y_train)
tree_grid.best_params_
y_tree_grid_pred = tree_grid.predict(x_test)
raport_log = classification_report(y_test, y_tree_grid_pred, labels=[0,1])
print(raport_log)
from sklearn.ensemble import GradientBoostingClassifier

grad_boost = GradientBoostingClassifier()
grad_boost.fit(x_train, y_train)
y_boost_pred = grad_boost.predict(x_test)
raport_log = classification_report(y_test, y_boost_pred, labels=[0,1])
print(raport_log)
param_grid = {'learning_rate':[0.05,0.1,0.2],
             'n_estimators':[50,100],'max_depth':[3,5,10]}
boost_grid = GridSearchCV(grad_boost, param_grid, scoring='accuracy')
boost_grid.fit(x_train, y_train)
boost_grid.best_params_
y_boost_grid_pred = boost_grid.predict(x_test)
raport_log = classification_report(y_test, y_boost_grid_pred, labels=[0,1])
print(raport_log)
df_copy['income'] = df_copy['income'].map({'<=50K':0, '>50K':1}).copy()
numeric_df = pd.concat([df_copy['fnlwgt'], df_copy['education.num'], 
                       df_copy['capital.gain'],df_copy['capital.loss'],
                       df_copy['hours.per.week'],df_copy['income']], axis=1, keys=['fnlwgt',
                                                           'education.num',
                                                           'capital.gain',
                                                           'capital.loss',
                                                           'hours.per.week','income'])
numeric_df.tail()
pd.value_counts(numeric_df['income'])
numeric_df.isnull().sum()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
numeric_df.hist(bins=20, figsize=(20,20))
plt.show()
corr_matrix = numeric_df.corr()
sns.heatmap(corr_matrix, xticklabels=corr_matrix.columns, yticklabels=corr_matrix.columns)
plt.show()
corr_matrix['income'].sort_values(ascending=False)
numeric_df['capital'] = (numeric_df['capital.gain'] - numeric_df['capital.loss'])
corr_matrix = numeric_df.corr()
corr_matrix['income'].sort_values(ascending=False)
from sklearn.model_selection import train_test_split
X_numeric = numeric_df.drop('income', axis=1)
y_numeric = numeric_df['income']
x_train_numeric, x_test_numeric, y_train_numeric, y_test_numeric = train_test_split(X_numeric, y_numeric, test_size=0.1)
x_train_numeric.shape, x_test_numeric.shape, y_train_numeric.shape, y_test_numeric.shape
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
log_reg.fit(x_train_numeric, y_train_numeric)
from sklearn.metrics import classification_report
y_log_pred = log_reg.predict(x_test_numeric)
raport_log = classification_report(y_test_numeric, y_log_pred, labels=[0,1])
print(raport_log)
from sklearn.ensemble import RandomForestClassifier

rand_forest = RandomForestClassifier()
rand_forest.fit(x_train_numeric, y_train_numeric)
y_random_pred = rand_forest.predict(x_test_numeric)
raport_forest = classification_report(y_test_numeric, y_random_pred, labels=[0,1])
print(raport_forest)
from sklearn.model_selection import GridSearchCV
param_grid = {"max_depth": [3, None],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
grid_search_forest = GridSearchCV(rand_forest, param_grid, scoring='accuracy')
grid_search_forest.fit(x_train_numeric, y_train_numeric)
grid_search_forest.best_params_
y_random_pred_grid = grid_search_forest.predict(x_test_numeric)
raport_forest_grid = classification_report(y_test_numeric, y_random_pred_grid, labels=[0,1])
print(raport_forest_grid)