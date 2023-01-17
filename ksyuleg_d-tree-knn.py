%matplotlib inline
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
import matplotlib
import seaborn as sns
from sklearn.metrics import log_loss
from itertools import combinations
from sklearn.metrics import log_loss
import itertools as it
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

sns.set()
df_train = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/train.csv')
df_test = pd.read_csv('/kaggle/input/ods-mlclass-dubai-2019-03-lecture3-hw/test.csv')
# Remove duplicated rows
print("Duplicates in train set: ",df_train.duplicated().sum())
df_train.drop_duplicates(keep="first", inplace = True)
df_test['target'] = np.nan
df = pd.concat([df_train, df_test])
print(df.shape)
cats = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
nums = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
for i in range(1, 4):
    print(i)
    for g in it.combinations(cats, i):
        df = pd.concat(
            [
                df, 
                df.groupby(list(g))[nums].transform('mean').rename(
                    columns=dict([(s, ':'.join(g) + '__' + s + '__mean') for s in nums])
                )
            ], 
            axis=1
        )
df.drop(columns=cats, inplace=True)
df.head()
cols = [c for c in df.columns if c != 'uid' and c != 'target']
df[cols] = StandardScaler().fit_transform(df[cols])
df_m = df[cols].corr()
df_m.head()
cor = {}
for c in cols:
    cor[c] = set(df_m.loc[c][df_m.loc[c] > 0.5].index) - {c}
    
len(cor)
for c in cols:
    if c not in cor:
        continue
    for s in cor[c]:
        if s in cor:
            cor.pop(s)
cols = list(cor.keys())
cols
model = KNeighborsClassifier(
    n_neighbors=100,
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    n_jobs=-1
)

model = model.fit(df.loc[df['target'].notna()][cols], df.loc[df['target'].notna()]['target'])
p = model.predict_proba(df.loc[df['target'].isna()][cols]) # test features
sns.distplot(p[:, 1])
df = pd.concat([df_train, df_test])
df.head()
df_tmp = df.loc[
    df['target'].notna()
].groupby(
    ['education']
)[
    'target'
].agg(['mean', 'std']).rename(
    columns={'mean': 'target_mean', 'std': 'target_std'}
).fillna(0.0).reset_index()

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
model = DecisionTreeClassifier(
    criterion='gini',
    splitter='best',
    max_depth=8,
    min_samples_split=42,
    min_samples_leaf=17
)

model = model.fit(df.loc[df['target'].notna()].drop(columns=['target']), df.loc[df['target'].notna()]['target'])
model.predict_proba(df.loc[df['target'].isna()].drop(columns=['target']).head())
p2 = model.predict_proba(df.loc[df['target'].isna()].drop(columns=['target']))[:, 1]
sns.distplot(p2)
p_f = 0.25 * p[:, 1] + 0.75 * p2

sns.distplot(p_f)
df_submit = pd.DataFrame({
    'uid': df.loc[df['target'].isna()]['uid'],
    'target': p_f
})