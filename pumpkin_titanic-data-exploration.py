import numpy as np
import pandas as pd
sample_submission = pd.read_csv('../input/titanic/gender_submission.csv')
test = pd.read_csv('../input/titanic/test.csv')
train = pd.read_csv('../input/titanic/train.csv')
test.shape, train.shape
test.head()
train.head()
train['Survived'].value_counts(dropna=False)
from sklearn import model_selection

kf = model_selection.StratifiedKFold(n_splits=5)

x = list(kf.split(train, train['Survived'].values))
[len(i) for i in x]
all_data = train.append(test, sort=True).fillna({'Survived': -1})
len(train), len(test), len(all_data)
def count_na(series):
    return series.isna().sum()
agg_funcs = ('nunique', count_na)

dv = 'Survived'

summary_df = all_data.aggregate(agg_funcs, dropna=False).transpose()

for dv_val in all_data[dv].unique():
    dv_val_data = (
        all_data
        .loc[lambda df: df[dv] == dv_val]
        .aggregate(agg_funcs, dropna=False).transpose()
    )

    # survived_data.columns = pd.MultiIndex.from_product([[f'Survived = {survived}'], survived_data.columns])
    summary_df[pd.MultiIndex.from_product([[f'{dv}_{dv_val}'], dv_val_data.columns])] = dv_val_data

summary_df['dtypes'] = all_data.dtypes
summary_df['cardinality'] = np.where(summary_df['nunique'] > 20, 'many', 'few')

summary_df = summary_df.sort_values(['dtypes', 'nunique']).drop(dv)
summary_df
# Cases to deal with
for _, row in summary_df[['dtypes', 'cardinality']].drop_duplicates().iterrows():
    print(
        f"{row['dtypes']} {row['cardinality']}:",
        summary_df.loc[
            lambda df: (df['dtypes'] == row['dtypes']) & (df['cardinality'] == row['cardinality'])
        ].index.tolist(),
    )