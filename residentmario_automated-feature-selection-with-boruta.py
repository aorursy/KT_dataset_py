import pandas as pd
import numpy as np
pd.set_option('max_columns', None)
kepler = pd.read_csv("../input/cumulative.csv")
kepler = (kepler
     .drop(['rowid', 'kepid'], axis='columns')
     .rename(columns={'koi_disposition': 'disposition', 'koi_pdisposition': 'predisposition'})
     .pipe(lambda df: df.assign(disposition=(df.disposition == 'CONFIRMED').astype(int), predisposition=(df.predisposition == 'CANDIDATE').astype(int)))
     .pipe(lambda df: df.loc[:, df.dtypes.values != np.dtype('O')])  # drop str columns
     .pipe(lambda df: df.loc[:, (df.isnull().sum(axis='rows') < 500).where(lambda v: v).dropna().index.values])  # drop columns with greater than 500 null values
     .dropna()
)

kepler_X = kepler.iloc[:, 1:]
kepler_y = kepler.iloc[:, 0]

kepler.head()
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, max_depth=5)

trans = BorutaPy(clf, random_state=42, verbose=2)
sel = trans.fit_transform(kepler_X.values, kepler_y.values)
trans.ranking_