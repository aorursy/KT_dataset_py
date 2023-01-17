import numpy as np

import pandas as pd

from matplotlib import pyplot as plt
# read in files

whr17 = pd.read_csv('../input/world-happiness/2017.csv', index_col='Country')

eii17 = pd.read_csv('../input/expat-insider-2017/Expat_Insider_Index_2017.csv', index_col='Country')
# drop unnecessary columns from world happiness report

whr17.drop(['Happiness.Rank', 'Whisker.high', 'Whisker.low'], axis=1, inplace=True)

# rename columns so that they can be more easily identified when compared with columns from other report

whr17.rename(columns={'Happiness.Score': 'WHR_Happiness', 'Economy..GDP.per.Capita.': 'WHR_GDPPC', 'Family': 'WHR_Family', 

                     'Health..Life.Expectancy.': 'WHR_Life_Expectancy', 'Freedom': 'WHR_Freedom', 'Generosity': 'WHR_Generosity',

                     'Trust..Government.Corruption.': 'WHR_Gov_Corruption', 'Dystopia.Residual': 'WHR_Dystopia_Residual'},

             inplace=True)

# created ranked version of whr17 to be joined with Expat insider files since all data in EI files is in terms of ranking

whr17ranked = whr17.rank(ascending=False)
# World Happiness Report joined with EI

whreii17 = whr17ranked.join(eii17, how='inner')
# re-ranking since a number of the rows from the WHR were lost in the join throwing off the values

whreii17 = whreii17.rank()
# Rank sum totals and overall rank columns added

whreii17['RankSum'] = whreii17.sum(axis=1)

whreii17['OverallRank'] = whreii17['RankSum'].rank()

eii17['RankSum'] = eii17.sum(axis=1)

eii17['OverallRank'] = eii17['RankSum'].rank()
# convert everything to integer type

whreii17 = whreii17.astype('int64')

eii17 = eii17.astype('int64')
# check for correlation between WHR happiness and other areas of the WHR

whr17.corr().loc['WHR_Happiness'].abs().sort_values(ascending=False)
# see what areas of the Expat Insider index have the highest correlation with areas of the WHR

whreii17.corr().loc[:'WHR_Dystopia_Residual', 'QualityofLife':].abs().unstack().sort_values(ascending=False).head(n=10)
# see what areas have the highest correlattion between happiness levels for expats

whreii17.corr().loc['PersonalHappiness'].abs().sort_values(ascending=False).head(n=10)
# graph PersonalHappiness and EaseofSettlingIn since the areas of highest correlation with 

# PersonalHapiness(FindingFriends, EaseofSettlingIn, FeelingWelcome, and Friendliness) are

# all contained within the EaseofSettlingIn group

whreii17[['PersonalHappiness', 'EaseofSettlingIn']].plot.bar(figsize=(20,10), grid=True)

plt.show()
# graph to show the horrible correlation between WHR happiness and expat happiness

whreii17.sort_values(by='PersonalHappiness')[['WHR_Happiness', 'PersonalHappiness']].plot.bar(figsize=(20,10), grid=True)

plt.show()
# check countries with best overall rank 

whreii17.loc[:, 'RankSum':].sort_values(by='OverallRank')
# check countries with best overall happiness between WHR and EI

whreii17['HappinessSum'] = whreii17[['WHR_Happiness', 'PersonalHappiness']].sum(axis=1)

whreii17['HappinessSumRank'] = whreii17['HappinessSum'].rank(method='first').astype('int64')

whreii17[['HappinessSum', 'HappinessSumRank']].sort_values(by='HappinessSumRank')