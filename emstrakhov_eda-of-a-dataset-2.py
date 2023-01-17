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
import matplotlib.pyplot as plt
sbm = pd.read_csv('/kaggle/input/softserve-ds-hackathon-2020/submission.csv')

sbm.head()
sbm.shape
sbm['EmployeeID'].nunique()
employees = pd.read_csv('/kaggle/input/softserve-ds-hackathon-2020/employees.csv', 

                        parse_dates=['HiringDate', 'DismissalDate'])

employees.head()
employees.shape
employees[employees['DismissalDate'].isna()].shape
history = pd.read_csv('/kaggle/input/softserve-ds-hackathon-2020/history.csv', 

                      parse_dates=['Date'])

history.head()
history.shape
history.info()
employees['EmployeeID'].nunique()
history['EmployeeID'].nunique()
len(set(employees['EmployeeID']) & set(history['EmployeeID']))
len(set(sbm['EmployeeID']) & set(history['EmployeeID']))
df = history.merge(employees, how='left', on='EmployeeID')

df.head()
df['diff'] = df['DismissalDate'].sub(df['Date']) / np.timedelta64(1, 'M')

df.head()
df.set_index('EmployeeID', inplace=True)

employees.set_index('EmployeeID', inplace=True)
df[~(employees['DismissalDate'].isna())].head(11)
df['target'] = (df['diff'] < 4).astype(int)

df.head()
train = df[ ~(df['DismissalDate'].isna()) ]

test = df[ df['DismissalDate'].isna() ]



print(train.shape, test.shape)
# df.drop(['DismissalDate', 'diff'], axis=1, inplace=True)



df['experience'] = df['Date'].sub(df['HiringDate']) / np.timedelta64(1, 'M')

df.head()
# df.drop(['Date', 'HiringDate'], axis=1, inplace=True)
df.columns.values
categorical = ['DevCenterID', 'SBUID', 'PositionID', 'PositionLevel', 

               'LanguageLevelID', 'CustomerID', 'ProjectID', 

               'CompetenceGroupID', 'FunctionalOfficeID', 'PaymentTypeId']

df_1 = pd.get_dummies(columns=categorical, data=df)

df_1.head()
cols_to_drop = ['Date', 'HiringDate', 'DismissalDate', 'diff']

train = df_1[ ~(df_1['DismissalDate'].isna()) ].drop(cols_to_drop, axis=1)

test = df_1[ df_1['DismissalDate'].isna() ].drop(cols_to_drop + ['target'], axis=1)
X = train.drop('target', axis=1)

y = train['target']
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 

                                                      test_size=0.3, random_state=1)
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

tree.fit(X_train, y_train)



y_pred = tree.predict(X_valid)
def fbeta_score(y_true, y_pred):

    from sklearn.metrics import precision_score, recall_score

    beta = 1.7

    p = precision_score(y_true, y_pred)

    r = recall_score(y_true, y_pred)

    return (1+beta**2) * p * r / ((beta**2) * p + r)
print(fbeta_score(y_valid, y_pred))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_valid, y_pred))
def plot_validation_curve(model_grid, param_name, params=None):

    # Рисуем валидационную кривую

    # По оси х --- значения гиперпараметров (param_***)

    # По оси y --- значения метрики (mean_test_score)



    results_df = pd.DataFrame(model_grid.cv_results_)

    

    if params == None:

        plt.plot(results_df['param_'+param_name], results_df['mean_test_score'])

    else:

        plt.plot(params, results_df['mean_test_score'])



    # Подписываем оси и график

    plt.xlabel(param_name)

    plt.ylabel('Test F1 score')

    plt.title('Validation curve')

    plt.show()
from sklearn.model_selection import GridSearchCV



tree_params = {'max_depth': np.arange(2, 10)}

tree_grid = GridSearchCV(tree, tree_params, cv=5, scoring='f1')

tree_grid.fit(X_train, y_train)
plot_validation_curve(tree_grid, 'max_depth')
tree_grid.best_score_
best_tree = tree_grid.best_estimator_

best_tree.fit(X, y)
y_test = best_tree.predict(test)

pd.Series(y_test).value_counts()
test.shape
train.shape
len(set(test.index.values))
test['target'] = y_test
group = test.reset_index().groupby('EmployeeID')['target'].mean()

group = pd.DataFrame(group).reset_index()
sbm_1 = sbm.merge(group, how='inner', on='EmployeeID')

sbm_1.head()
sbm_1['target'] = (sbm_1['target_y'] > 0.5).astype(int)

sbm_2 = sbm_1.drop(['target_x', 'target_y'], axis=1)

sbm_2.head()
sbm_2['target'].value_counts()
sbm_2.to_csv('my_submission.csv', index=False)