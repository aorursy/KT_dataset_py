# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split, StratifiedKFold

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
print(df_train.shape)

df_test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
print(df_test.shape)
df_train.head()
df_train.groupby(['workclass'])\
  .agg({'target': [np.mean, np.min, np.max, 'count']})\
  .sort_values(('target','mean'), ascending=True)
df_train.groupby(['education','education-num'])\
  .agg({'target': [np.mean, np.min, np.max, 'count']})\
  .sort_values(('education-num'), ascending=True)
df_train.groupby(['relationship'])\
  .agg({'target': [np.mean, np.min, np.max, 'count']})\
  .sort_values(('target','mean'), ascending=True)

df_test['target'] = np.nan

df = pd.concat([df_train, df_test])
df['capital-net-change'] = df['capital-gain'] - df['capital-loss']

# print(df.shape)
# df.head()
# df.tail()
# df.dtypes
df[df['education-num'] < 9.0].loc[:,('education')] = 'Below College'
df.drop(columns=['education-num'])
# df_tmp = df.loc[
#     df['target'].notna()
# ].groupby(
#     ['education-native-country-workclass-combo']
# )[
#     'target'
# ].agg(['mean', 'std']).rename(
#     columns={'mean': 'target_mean', 'std': 'target_std'}
# ).fillna(0.0).reset_index()

# df_tmp.head()
# df = pd.merge(
#     df,
#     df_tmp,
#     how='left',
#     on=['education-native-country-workclass-combo']
# )

# df.shape
# df['target_mean'] = df['target_mean'].fillna(0.0)
# df['target_std'] = df['target_std'].fillna(0.0)
df['sex'] = df['sex'].replace(' Male', 0)
df['sex'] = df['sex'].replace(' Female', 1)
# pd.get_dummies(df['education'])
df = pd.get_dummies(
    df, 
    columns=[c for c in df_train.columns if df_train[c].dtype == 'object']
)
# df = pd.get_dummies(
#     df, 
#     columns=['education-native-country-workclass-combo']
# )
df.head()
X_train, X_test, y_train, y_test = train_test_split(df.loc[df['target'].notna()].drop(columns=['target']),
                                                    df.loc[df['target'].notna()]['target'],
                                                    test_size=0.33,
                                                    random_state=17)
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=8,
    min_samples_split=42,
    min_samples_leaf=17
)
parameter_grid = {
    'max_depth': range(5, 15),
    'max_features': range(1, 9),
    'min_samples_split': [35, 40, 45, 50],
    'min_samples_leaf': [5, 10, 15, 20],
}
grid_search = GridSearchCV(model, param_grid=parameter_grid, cv=5)
grid_search.fit(X_train, y_train)

# model = model.fit(df.loc[df['target'].notna()].drop(columns=['target']), df.loc[df['target'].notna()]['target'])
# model.predict_proba(df.loc[df['target'].isna()].drop(columns=['target']).head())
p = grid_search.predict_proba(df.loc[df['target'].isna()].drop(columns=['target']))[:, 1]
# %matplotlib inline
# import matplotlib
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# sns.set_style("dark")
# sns.distplot(p)
df_submit = pd.DataFrame({
    'uid': df.loc[df['target'].isna()]['uid'],
    'target': p
})
df_submit.to_csv('/kaggle/working/submit.csv', index=False)
#!head /kaggle/working/submit.csv
from sklearn.metrics import log_loss
log_loss(y_test, grid_search.predict_proba(X_test)[:, 1])

