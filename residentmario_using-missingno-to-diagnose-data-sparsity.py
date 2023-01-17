import pandas as pd

races = pd.read_csv("../input/tips.csv", encoding='latin1')

races.head(3)
import numpy as np

odds = (races

            .pivot(index='UID', columns='Tipster', values='Odds')

            .assign(ID=races.ID)

            .fillna(0)

            .groupby('ID')

            .sum()

            .assign(Win=races.Result.map(lambda v: v == 'Win'))

            .replace(0, np.nan))
odds.head()
import missingno as msno

msno.matrix(odds.drop('Win', axis='columns').sample(500))
msno.bar(odds.drop('Win', axis='columns'))
msno.heatmap(odds.drop('Win', axis='columns'))
msno.dendrogram(odds.drop('Win', axis='columns'))
pred = odds.loc[:, ['Tipster A1', 'Tipster B1', 'Tipster E', 'Tipster X']].dropna()

y = odds.loc[pred.index, 'Win']