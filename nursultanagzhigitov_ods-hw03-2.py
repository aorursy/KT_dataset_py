# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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
df_train.corr()
df_test['target'] = np.nan

df = pd.concat([df_train, df_test])

print(df.shape)
df.dtypes
df_tmp = df.loc[
    df['target'].notna()
].groupby(
    ['education']
)[
    'target'
].agg(['mean', 'std']).rename(
    columns={'mean': 'target_mean', 'std': 'target_std'}
).fillna(0.0).reset_index()

df_tmp.head()
df = pd.merge(
    df,
    df_tmp,
    how='left',
    on=['education']
)

df.shape
df['target_mean'] = df['target_mean'].fillna(0.0)
df['target_std'] = df['target_std'].fillna(0.0)
pd.get_dummies(df['workclass'])
df = pd.get_dummies(
    df, 
    columns=[c for c in df_train.columns if df_train[c].dtype == 'object']
)
df.head()
df['hours-per-week'].nunique()
df['capital-gain']
df['age']
df_tmp2 = df.loc[
    df['target'].notna()
].groupby(
    ['hours-per-week']
)[
    'target'
].agg(['mean', 'std']).rename(
    columns={'mean': 'target_mean_hours_per_week', 'std': 'target_std_hours_per_week'}
).fillna(0.0).reset_index()

df_tmp2.head()
df = pd.merge(
    df,
    df_tmp2,
    how='left',
    on=['hours-per-week']
)

df.shape
df['target_mean_hours_per_week'] = df['target_mean_hours_per_week'].fillna(0.0)
df['target_std_hours_per_week'] = df['target_std_hours_per_week'].fillna(0.0)
df_tmp3 = df.loc[
    df['age'].notna()
].groupby(
    ['age']
)[
    'target'
].agg(['mean', 'std']).rename(
    columns={'mean': 'target_mean_age', 'std': 'target_std_age'}
).fillna(0.0).reset_index()

df_tmp3.head()
df_tmp3.head(30)
df = pd.merge(
    df,
    df_tmp3,
    how='left',
    on=['age']
)

df.shape
df['target_mean_age'] = df['target_mean_age'].fillna(0.0)
df['target_std_age'] = df['target_std_age'].fillna(0.0)
df_tmp4 = df.loc[
    df['capital-gain'].notna()
].groupby(
    ['capital-gain']
)[
    'target'
].agg(['mean', 'std']).rename(
    columns={'mean': 'target_mean_capital-gain', 'std': 'target_std_capital-gain'}
).fillna(0.0).reset_index()

df_tmp4.head()
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=6,
    min_samples_split=42,
    min_samples_leaf=17
)

model = model.fit(df.loc[df['target'].notna()].drop(columns=['target']), df.loc[df['target'].notna()]['target'])
model.predict_proba(df.loc[df['target'].isna()].drop(columns=['target']).head())
p = model.predict_proba(df.loc[df['target'].isna()].drop(columns=['target']))[:, 1]
%matplotlib inline
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("dark")
sns.distplot(p)
df_submit = pd.DataFrame({
    'uid': df.loc[df['target'].isna()]['uid'],
    'target': p
})
df_submit.to_csv('/kaggle/working/submit.csv', index=False)
!head /kaggle/working/submit.csv
