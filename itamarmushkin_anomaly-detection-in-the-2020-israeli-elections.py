import pandas as pd

import numpy as np

from sklearn.ensemble import IsolationForest



filename = '/kaggle/input/israeli-elections-2015-2013/votes per booth 2020.csv'

encoding = 'iso_8859_8'



df_raw = pd.read_csv(filename, encoding=encoding)

df_raw.dropna(axis=1, how='all', inplace=True)

df_raw.head()
df = df_raw.copy()



oversight_columns = ['סמל ועדה', 'ברזל', 'שופט', 'ריכוז']

df.drop(oversight_columns, axis=1, inplace=True)



index_columns = ['שם ישוב', 'סמל ישוב', 'קלפי']

metadata_columns = ['בזב', 'מצביעים', 'פסולים', 'כשרים']



party_columns = df.columns.difference(pd.Index(index_columns+metadata_columns))

pd.testing.assert_series_equal(df['כשרים'], df[party_columns].sum(axis=1).rename('כשרים'))

#df.set_index(index_columns, inplace=True) - will set index after feature engineering

df.head()
df['turnout'] = (df['מצביעים'] / df['בזב']).replace(np.inf, -1)

df['percent_invalid'] = df['פסולים'] / df['מצביעים']



df["total_voting_booths"] = df.groupby(["סמל ישוב"])["קלפי"].transform("size")

df["booth_per_capita"] = df["total_voting_booths"].div(df["בזב"]).replace(np.inf, -1)



df["max_party_vote"] = df[party_columns].max(axis=1)

df["max_party_ratio"] =df['max_party_vote'].div(df['כשרים'], axis=0)



df[party_columns] = df[party_columns].div(df["כשרים"], axis=0)



df.set_index(index_columns, inplace=True)



assert(not(df.replace([np.inf, -np.inf], np.nan).isna().any().any()))

df.head()
party_votes = df[party_columns].mul(df['כשרים'], axis=0).sum() / df['כשרים'].sum()

party_votes.sort_values(ascending=False)

threshold = 0.001

major_parties = party_votes[party_votes >= threshold].index

minor_parties = party_votes[party_votes < threshold].index

print('the major parties (those with >{} of votes) are'.format(threshold))

print(major_parties.to_list())

df.drop(minor_parties, inplace=True, axis=1)
model = IsolationForest(n_estimators=140,max_samples=600).fit(df)

predictions = pd.Series(index=df.index, data=model.decision_function(df), name='anomaly score').sort_values(ascending=False)

predictions.to_frame()
predictions.tail(10)
# Saving the results for further use



df_raw.set_index(index_columns).join(predictions).to_csv('anomalous_booths_2020_2.csv', encoding=encoding)