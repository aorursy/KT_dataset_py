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
df_test['target'] = np.nan

df = pd.concat([df_train, df_test])

print(df.shape)
df.head()
df.tail()
df.dtypes
pd.get_dummies(df['workclass'])
df = pd.get_dummies(
    df, 
    columns=[c for c in df_train.columns if df_train[c].dtype == 'object']
)
df.head()

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=8,
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



