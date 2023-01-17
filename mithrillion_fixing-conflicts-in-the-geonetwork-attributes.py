import numpy as np
import pandas as pd
import random
import re

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
dat = pd.read_pickle("../input/gstore-revenue-data-preprocessing/train.pkl")
geo_colnames = [c for c in dat.columns if re.match(r'geoNetwork', c) is not None]
geo_colnames
pure_geo_columns = [c for c in geo_colnames if c != 'geoNetwork.networkDomain']
for c in pure_geo_columns:
    dat.loc[:, c] = dat[c].cat.add_categories('N/A').fillna('N/A')
selected = dat.loc[:, dat.columns.isin(pure_geo_columns)].copy()
selected.groupby(pure_geo_columns).size().reset_index().head(30)
country_part = selected.groupby(
    ['geoNetwork.continent', 'geoNetwork.country',
     'geoNetwork.subContinent']).size().reset_index()

country_part.head(20)
country_part[country_part['geoNetwork.country'].duplicated(keep=False)]
city_part = selected.groupby(
    ['geoNetwork.city', 'geoNetwork.metro',
     'geoNetwork.region']).size().reset_index()

city_part.head(20)
city_part[city_part['geoNetwork.city'].duplicated(keep=False) & (city_part['geoNetwork.city'] != 'N/A')]
city_part.shape
city_part.groupby(['geoNetwork.city', 'geoNetwork.region']).size().sort_values(ascending=False).head(10)
pairs = [('Colombo', 'Western Province'), ('Doha', 'Doha'),
         ('Guatemala City', 'Guatemala Department'), ('Hanoi', 'Hanoi'),
         ('Minsk', 'Minsk Region'), ('Nairobi', 'Nairobi County'), ('Tbilisi',
                                                                    'Tbilisi')]

for c, r in pairs:
    dat.loc[(dat['geoNetwork.city'] == c) &
            (dat['geoNetwork.region'] == 'N/A'), 'geoNetwork.region'] = r
city_part[city_part['geoNetwork.city'] == 'N/A']
selected = dat.loc[:, dat.columns.isin(pure_geo_columns)].copy()
cc = selected.groupby(['geoNetwork.city', 'geoNetwork.region','geoNetwork.country']).size().reset_index()

cc.loc[(cc['geoNetwork.city'].duplicated(keep=False) &
        (cc['geoNetwork.city'] != 'N/A'))
       | (cc['geoNetwork.region'].duplicated(keep=False) & (
           (cc['geoNetwork.city'] == 'N/A') &
           (cc['geoNetwork.region'] != 'N/A')))].head(30)
most_common = dat.groupby([
    'geoNetwork.city', 'geoNetwork.region'
])['geoNetwork.country'].apply(lambda x: x.mode()).reset_index()
most_common.head()
for idx, row in most_common.iterrows():
    dat.loc[(dat['geoNetwork.city'] == row['geoNetwork.city']) &
            (dat['geoNetwork.region'] == row['geoNetwork.region']) &
            ((dat['geoNetwork.city'] != 'N/A') |
             ((dat['geoNetwork.region'] != 'N/A'))
             ), 'geoNetwork.country'] = row['geoNetwork.country']

selected = dat.loc[:, dat.columns.isin(pure_geo_columns)].copy()
cc = selected.groupby(
    ['geoNetwork.city', 'geoNetwork.region',
     'geoNetwork.country']).size().reset_index()
cc.head(30)
dat2 = pd.read_pickle("../input/gstore-revenue-data-preprocessing/train.pkl")
for c in pure_geo_columns:
    dat2.loc[:, c] = dat2[c].cat.add_categories('N/A').fillna('N/A')
    
selected = dat2.loc[:, pure_geo_columns].copy()
key_attributes = ['geoNetwork.city', 'geoNetwork.region', 'geoNetwork.country']
selected[key_attributes].describe()
enc = OneHotEncoder()
transformed = enc.fit_transform(selected[key_attributes].apply(lambda x: x.cat.codes))
N, D = transformed.shape
N, D
clf = RandomForestClassifier()
clf.fit(transformed[:, :-222], selected['geoNetwork.country'].cat.codes)
pred = clf.predict(transformed[:, :-222])
is_anomaly = pred != selected['geoNetwork.country'].cat.codes
anomaly_cases = selected[is_anomaly][(selected['geoNetwork.city'] != 'N/A') | (
    selected['geoNetwork.region'] != 'N/A')][key_attributes]
anomaly_cases.head(30)
certainty = clf.predict_proba(transformed[:, :-222])
uncertain_idx = np.max(certainty, axis=1) < 0.95
uncertain_cases = selected[key_attributes][uncertain_idx]
uncertain_cases.head()
uncertain_cases[(uncertain_cases['geoNetwork.city'] != 'N/A')
                & (uncertain_cases['geoNetwork.region'] != 'N/A')].groupby(key_attributes).size().reset_index()
not_na_idx = (selected['geoNetwork.city'] !=
              'N/A') | (selected['geoNetwork.region'] != 'N/A')

target = pd.Categorical.from_codes(
    pred, categories=selected['geoNetwork.country'].cat.categories)

dat2.loc[not_na_idx, 'geoNetwork.country'] = target[not_na_idx]
diff_idx = np.any(dat[key_attributes] != dat2[key_attributes], axis=1)
pd.concat([dat[key_attributes][diff_idx], dat2[key_attributes][diff_idx]], axis=1).head(10)
diff_idx = np.any(dat[key_attributes] != dat2[key_attributes], axis=1) & (dat2['geoNetwork.region'] != 'N/A')
pd.concat([dat[key_attributes][diff_idx], dat2[key_attributes][diff_idx]], axis=1).head(10)
for c in pure_geo_columns:
    dat.loc[:, c] = dat[c].cat.remove_categories('N/A')

dat.to_pickle("manual_geo_fix.pkl")

for c in pure_geo_columns:
    dat2.loc[:, c] = dat2[c].cat.remove_categories('N/A')

dat2.to_pickle("classifier_geo_fix.pkl")
original = pd.read_pickle("../input/gstore-revenue-data-preprocessing/train.pkl")
original[is_anomaly & (~pd.isna(original['geoNetwork.networkDomain'])) &
         (~pd.isna(original['geoNetwork.city'])
          | ~pd.isna(original['geoNetwork.region']))][
              key_attributes + ['geoNetwork.networkDomain']].head(30)
original[key_attributes + ['fullVisitorId']].dropna().groupby('fullVisitorId')[[
    'geoNetwork.city', 'geoNetwork.country'
]].nunique().mean()
original[key_attributes + ['sessionId']].dropna().groupby('sessionId')[[
    'geoNetwork.city', 'geoNetwork.country'
]].nunique().mean()
