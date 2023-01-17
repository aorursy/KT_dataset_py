import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns 

import warnings

warnings.filterwarnings("ignore")

train = pd.read_csv('../input/train_V2.csv')
#抓極端值 wlakdistance

data = train.copy()

#walkDistance

print(np.percentile(data['walkDistance'],99))

print(len(data['walkDistance']))

sq1 = 'walkDistance > 4396 '

sub = train.query(sq1)

print(sq1, '\n count:', len(sub), ' winPlacePerc:', round(sub['winPlacePerc'].mean(),3))
#rideDistance

print(np.percentile(data['rideDistance'],99))

print(len(data['rideDistance']))

sq2 = 'rideDistance > 6966'

sub = train.query(sq2)

print(sq2, '\n count:', len(sub), ' winPlacePerc:', round(sub['winPlacePerc'].mean(),3))
#boosts

print(np.percentile(data['boosts'],99))

print(len(data['boosts']))

sq3 = 'boosts > 7.0'

sub = train.query(sq3)

print(sq3, '\n count:', len(sub), ' winPlacePerc:', round(sub['winPlacePerc'].mean(),3))
#weaponsAcquired

print(np.percentile(data['weaponsAcquired'],99))

print(len(data['weaponsAcquired']))

sq4 = 'weaponsAcquired > 10.0'

sub = train.query(sq4)

print(sq4, '\n count:', len(sub), ' winPlacePerc:', round(sub['winPlacePerc'].mean(),3))
#damageDealt

print(np.percentile(data['damageDealt'],99))

print(len(data['damageDealt']))

sq5 = 'damageDealt > 776.2'

sub = train.query(sq5)

print(sq5, '\n count:', len(sub), ' winPlacePerc:', round(sub['winPlacePerc'].mean(),3))
#kills

print(np.percentile(data['kills'],99))

print(len(data['kills']))

sq6 = 'kills > 7'

sub = train.query(sq6)

print(sq6, '\n count:', len(sub), ' winPlacePerc:', round(sub['winPlacePerc'].mean(),3))
#heals

print(np.percentile(data['heals'],99))

print(len(data['heals']))

sq7 = 'heals > 12'

sub = train.query(sq7)

print(sq7, '\n count:', len(sub), ' winPlacePerc:', round(sub['winPlacePerc'].mean(),3))
data = train.copy()

data = data[data['walkDistance'] < train['walkDistance'].quantile(0.99)]

data = data[data['rideDistance'] < train['rideDistance'].quantile(0.99)]

data = data[data['boosts'] < train['boosts'].quantile(0.99)]

data = data[data['weaponsAcquired'] < train['weaponsAcquired'].quantile(0.99)]

data = data[data['damageDealt'] < train['damageDealt'].quantile(0.99)]

data = data[data['kills'] < train['kills'].quantile(0.99)]

data = data[data['heals'] < train['heals'].quantile(0.99)]

print(round(data['winPlacePerc'].describe(),3))
#行走距離分析 walkDistance

data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

A1 = (data["matchType"] == "solo")

A2 = (data['walkDistance']> 4396)

data=data[(A1 & A2)]    

print(round(data['winPlacePerc'].describe(),3))



data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

B1 = (data["matchType"] == "duo")

B2 = (data['walkDistance']> 4396)

data=data[(B1 & B2)]    

print(round(data['winPlacePerc'].describe(),3))



data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

C1 = (data["matchType"] == "squad")

C2 = (data['walkDistance']> 4396)

data=data[(C1 & C2)]    

print(round(data['winPlacePerc'].describe(),3))
#行駛距離分析 rideDistance

data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

A1 = (data["matchType"] == "solo")

A2 = (data['rideDistance']> 6966)

data=data[(A1 & A2)]    

print(round(data['winPlacePerc'].describe(),3))



data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

B1 = (data["matchType"] == "duo")

B2 = (data['rideDistance']> 6966)

data=data[(B1 & B2)]    

print(round(data['winPlacePerc'].describe(),3))



data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

C1 = (data["matchType"] == "squad")

C2 = (data['rideDistance']> 6966)

data=data[(C1 & C2)]    

print(round(data['winPlacePerc'].describe(),3))
#輔助道具與傷害造成交互作用分析

data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

A1 = (data["matchType"] == "solo")

A2 = (data['boosts']> 7.0)

A3 = (data['damageDealt']> 776.2)

data=data[(A1 & A2 & A3)]    

print(round(data['winPlacePerc'].describe(),3))



data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

B1 = (data["matchType"] == "duo")

B2 = (data['boosts']> 7.0)

B3 = (data['damageDealt']> 776.2)

data=data[(B1 & B2 & B3)]    

print(round(data['winPlacePerc'].describe(),3))



data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

C1 = (data["matchType"] == "squad")

C2 = (data['boosts']> 7.0)

C3 = (data['damageDealt']> 776.2)

data=data[(C1 & C2 & C3)]    

print(round(data['winPlacePerc'].describe(),3))
#所有極端值交互作用分析 99%

data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

A1 = (data["matchType"] == "solo")

A2 = (data['boosts']> 7.0)

A3 = (data['damageDealt']> 776.2)

A4 = (data['walkDistance']> 4396)

A5 = (data['rideDistance']> 6966)

A6 = (data['kills']> 7)

A7 = (data['weaponsAcquired']> 10.0)

A8 = (data['heals']> 12)

data=data[(A1 & A2 & A3 & A4 & A5 & A6 &  A8)]    

print(round(data['winPlacePerc'].describe(),3))



data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

B1 = (data["matchType"] == "duo")

B2 = (data['boosts']> 7.0)

B3 = (data['damageDealt']> 776.2)

B4 = (data['walkDistance']> 4396)

B5 = (data['rideDistance']> 6966)

B6 = (data['kills']> 7)

B7 = (data['weaponsAcquired']> 10.0)

B8 = (data['heals']> 12)

data=data[(B1 & B2 & B3 & B4 & B5 & B6 &  B8)] 

print(round(data['winPlacePerc'].describe(),3))



data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

C1 = (data["matchType"] == "squad")

C2 = (data['boosts']> 7.0)

C3 = (data['damageDealt']> 776.2)

C4 = (data['walkDistance']> 4396)

C5 = (data['rideDistance']> 6966)

C6 = (data['kills']> 7)

C7 = (data['weaponsAcquired']> 10.0)

C8 = (data['heals']> 12)

data=data[(C1 & C2 & C3 & C4 & C5 & C6 &  C8)]    

print(round(data['winPlacePerc'].describe(),3))
#所有極端值交互作用分析 99%

data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

A1 = (data["matchType"] == "solo")

A2 = (data['boosts']> 7.0)

A3 = (data['damageDealt']> 776.2)

A4 = (data['walkDistance']> 4396)

A5 = (data['rideDistance']> 6966)

A6 = (data['kills']> 7)

#A7 = (data['weaponsAcquired']> 10.0)

A8 = (data['heals']> 12)

data=data[(A1 & A2 & A3 & A4 & A5 & A6 &  A8)]    

print(round(data['winPlacePerc'].describe(),3))



data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

B1 = (data["matchType"] == "duo")

B2 = (data['boosts']> 7.0)

B3 = (data['damageDealt']> 776.2)

B4 = (data['walkDistance']> 4396)

B5 = (data['rideDistance']> 6966)

B6 = (data['kills']> 7)

#B7 = (data['weaponsAcquired']> 10.0)

B8 = (data['heals']> 12)

data=data[(B1 & B2 & B3 & B4 & B5 & B6 &  B8)] 

print(round(data['winPlacePerc'].describe(),3))



data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

C1 = (data["matchType"] == "squad")

C2 = (data['boosts']> 7.0)

C3 = (data['damageDealt']> 776.2)

C4 = (data['walkDistance']> 4396)

C5 = (data['rideDistance']> 6966)

C6 = (data['kills']> 7)

#C7 = (data['weaponsAcquired']> 10.0)

C8 = (data['heals']> 12)

data=data[(C1 & C2 & C3 & C4 & C5 & C6 &  C8)]    

print(round(data['winPlacePerc'].describe(),3))
data = train.copy()

#walkDistance

print(np.percentile(data['walkDistance'],95))



#rideDistance

print(np.percentile(data['rideDistance'],95))



#boosts

print(np.percentile(data['boosts'],95))



#weaponsAcquired

print(np.percentile(data['weaponsAcquired'],95))



#damageDealt

print(np.percentile(data['damageDealt'],95))



#kills

print(np.percentile(data['kills'],95))



#heals

print(np.percentile(data['heals'],95))
#所有極端值交互作用分析 95%

data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

A1 = (data["matchType"] == "solo")

A2 = (data['boosts']> 5.0)

A3 = (data['damageDealt']> 459.1)

A4 = (data['walkDistance']> 3396)

A5 = (data['rideDistance']> 4048)

A6 = (data['kills']> 4)

A7 = (data['weaponsAcquired']> 8.0)

A8 = (data['heals']> 7)

data=data[(A1 & A2 & A3 & A4 & A5 & A6 &  A8)]    

print(round(data['winPlacePerc'].describe(),3))



data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

B1 = (data["matchType"] == "duo")

B2 = (data['boosts']> 5.0)

B3 = (data['damageDealt']> 459.1)

B4 = (data['walkDistance']> 3396)

B5 = (data['rideDistance']> 4048)

B6 = (data['kills']> 4)

B7 = (data['weaponsAcquired']> 8.0)

B8 = (data['heals']> 7)

data=data[(B1 & B2 & B3 & B4 & B5 & B6 &  B8)] 

print(round(data['winPlacePerc'].describe(),3))



data = train.copy()

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'

data['matchType'] = data['matchType'].apply(mapper)

C1 = (data["matchType"] == "squad")

C2 = (data['boosts']> 5.0)

C3 = (data['damageDealt']> 459.1)

C4 = (data['walkDistance']> 3396)

C5 = (data['rideDistance']> 4048)

C6 = (data['kills']> 4)

C7 = (data['weaponsAcquired']> 8.0)

C8 = (data['heals']> 7)

data=data[(C1 & C2 & C3 & C4 & C5 & C6 &  C8)]    

print(round(data['winPlacePerc'].describe(),3))