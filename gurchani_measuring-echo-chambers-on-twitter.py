import pandas as pd

importantFeatureFollowers = pd.read_csv('../input/polarizationfollowersoffeatureslatest/polarizationinquiryfollowersbackup (1).csv')

importantFeatureFollowers['follower'] = importantFeatureFollowers['follower'].astype(str)

featureImportanceWithParty = pd.read_csv('../input/featureswithimportancescore/feature-importance-party1.csv')

combineNetworkWithimportance = importantFeatureFollowers.set_index('id').join(featureImportanceWithParty.set_index('id'))
import gc

del importantFeatureFollowers

del featureImportanceWithParty

gc.collect()

howPolitisized = combineNetworkWithimportance.groupby(['follower'])['importance'].count().astype('int8')

howPolitisized.head()
emNetwork = combineNetworkWithimportance[combineNetworkWithimportance['party'] == 'Em']

emImportance = emNetwork.groupby(['follower'])['importance'].sum().astype('float32')

emImportance.head()
combiningEverything = pd.concat([howPolitisized, emImportance], axis=1)

print(combiningEverything.head())

del emImportance

del howPolitisized

gc.collect()
psNetwork = combineNetworkWithimportance[combineNetworkWithimportance['party'] == 'Ps']

psImportance = psNetwork.groupby(['follower'])['importance'].sum().astype('float32')

psImportance.head()
combiningEverything = pd.concat([combiningEverything, psImportance], axis = 1)

del psImportance

gc.collect()

combiningEverything.head()
lrNetwork = combineNetworkWithimportance[combineNetworkWithimportance['party'] == 'Lr']

lrImportance = lrNetwork.groupby(['follower'])['importance'].sum().astype('float32')
combiningEverything = pd.concat([combiningEverything, lrImportance], axis = 1)

del lrImportance

gc.collect()

combiningEverything.head()
fnNetwork = combineNetworkWithimportance[combineNetworkWithimportance['party'] == 'Fn']

fnImportance = fnNetwork.groupby(['follower'])['importance'].sum().astype('float32')
combiningEverything = pd.concat([combiningEverything, fnImportance], axis = 1)

del fnImportance

gc.collect()

combiningEverything.head()
fiNetwork = combineNetworkWithimportance[combineNetworkWithimportance['party'] == 'Fi ']

fiImportance = fiNetwork.groupby(['follower'])['importance'].sum().astype('float32')
combiningEverything = pd.concat([combiningEverything, fiImportance], axis = 1)



del fiImportance

gc.collect()

combiningEverything.head()
del combineNetworkWithimportance

gc.collect()
combiningEverything.columns = ['PolitisizationScore', 'EmScore', 'PsScore', 'LrScore', 'FnScore', 'FiScore']

combiningEverything.fillna(0, inplace = True)

combiningEverything.head()
combiningEverything['Lr/total'] = combiningEverything['LrScore']/combiningEverything['PolitisizationScore']

combiningEverything['Em/total'] = combiningEverything['EmScore']/combiningEverything['PolitisizationScore']

combiningEverything['Ps/total'] = combiningEverything['PsScore']/combiningEverything['PolitisizationScore']

combiningEverything['Fn/total'] = combiningEverything['FnScore']/combiningEverything['PolitisizationScore']

combiningEverything['Fi/total'] = combiningEverything['FiScore']/combiningEverything['PolitisizationScore']
combiningEverything[combiningEverything['PolitisizationScore'] > 10].describe
morePolitisized = combiningEverything[combiningEverything['PolitisizationScore'] > 10]
import scipy.stats.stats as stats

print(stats.describe(morePolitisized['EmScore']))

print(stats.describe(morePolitisized['PsScore']))

print(stats.describe(morePolitisized['LrScore']))

print(stats.describe(morePolitisized['FnScore']))

print(stats.describe(morePolitisized['FiScore']))
combiningEverything.fillna(0, inplace = True)
print(stats.describe(combiningEverything['PolitisizationScore']))

combiningEverything[combiningEverything['PolitisizationScore'] > 0.03].shape

import scipy.stats.stats as stats

#print(stats.describe(combiningEverything['PolitisizationScore']))

#print(stats.scoreatpercentile(combiningEverything['PolitisizationScore'],  99.9))

print(stats.describe(combiningEverything['Lr/total']))

print(stats.describe(combiningEverything['Em/total']))

print(stats.describe(combiningEverything['Ps/total']))

print(stats.describe(combiningEverything['Fn/total']))

print(stats.describe(combiningEverything['Fi/total']))