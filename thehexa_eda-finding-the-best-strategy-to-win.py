# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
# Get the data file path
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Read train set to dataframe
main_df = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
# Data Exploration
print(main_df.shape)
main_df.head()
# Basic information about dataset
main_df.info()
# Number of players with percentile greater or equal to 90% 
print("# of players in top 10 places: ", sum(main_df['winPlacePerc'] >= 0.9))

# Number of players with percentile less than or equal to 10%
print("# of players in last 10 places: ", sum(main_df['winPlacePerc'] <= 0.1))
# Filtering only columns we are interested in
main_df_subset =  main_df[['assists', 'boosts', 'damageDealt', 'DBNOs', 'heals', 'kills', 'revives', 
                           'rideDistance', 'walkDistance', 'swimDistance', 'matchDuration', 'weaponsAcquired',
                                               'winPlacePerc']]
# Pearson correlation coefficient
corr = main_df_subset.corr()

sns.heatmap(corr, fmt='.1f', annot=True,
        xticklabels=corr.columns,
        yticklabels=corr.columns)
all_pairs = corr.unstack().sort_values(ascending=False).drop_duplicates()

print("Pairs with correlation >= 0.4:")
all_pairs[all_pairs >= 0.4]
# Jointplot of walkdistance vs winplace percentile 
sns.jointplot(x='winPlacePerc', y='walkDistance', data=main_df_subset, color='green')
# Jointplot of rideDistane vs winplace percentile 
sns.jointplot(x='winPlacePerc', y='rideDistance', data=main_df_subset, color='green')
# Lineplot of swimdistance vs winplace percentile 
sns.lineplot(x='winPlacePerc', y='swimDistance', data=main_df_subset)
main_df_subset_top10 = main_df_subset[main_df_subset.winPlacePerc >= 0.9]
print("Average # of kills by players who came in top10 percentile: ", round(np.mean(main_df_subset_top10.kills)))
# Dividing kills into 7 different bins for better representation
main_df_subset['kills_bin'] = pd.cut(main_df_subset['kills'], [-1, 0, 3, 5, 7, 10, 15, 100], 
                                            labels=['0 kills','1-3 kills', '3-5 kills', '5-7 kills',
                                                    '7-10 kills', '10-15 kills', '15+ kills'])

# Boxplot of winPlacePerc vs kills
plt.figure(figsize=(15,8))
sns.boxplot(x='kills_bin', y='winPlacePerc', data=main_df_subset)
# Boosts vs winPlacePerc
sns.jointplot(x='winPlacePerc', y='boosts', data=main_df_subset, color='crimson')
print("Average # of boosts used by players in top10 percentile: ", round(np.mean(main_df_subset_top10.boosts)))
# Boosts vs kills
plt.figure(figsize=(15,8))
sns.boxplot(x='kills_bin', y='boosts', data=main_df_subset)
# weapons acquired vs win place percetile 
sns.jointplot(x='winPlacePerc', y='weaponsAcquired', data=main_df_subset, color='teal')
# heals vs win place percetile 
sns.jointplot(x='winPlacePerc', y='heals', data=main_df_subset, color='maroon')
# 1) Max DBNOs
print("Sir knocks a lot! - Max DBNOs:", max(main_df.DBNOs))

# 2) Max assists in single match
print("Best assistant! - Max assists in a match:", max(main_df.assists))

# 3) Max number of Boosts
print("On a pill! - Max number of boosts:", max(main_df.boosts))

# 4) Max damage dealt
print("The berserker! - Max damage dealt:", max(main_df.damageDealt))

# 5) Max headshot kills
print("The Sniper! - Max number of headshot kills:", max(main_df.headshotKills))

# 6) Max heals
print("The doctor! - Max number of health packs used in a match:", max(main_df.heals))

# 7) Max kills
print("The killing machine! - Max number of kills in a match:", max(main_df.kills))

# 8) Max match duration
print("Camper! - Longest match ever: %f mins"%(max(main_df.matchDuration)/60))

# 9) Most revives
print("Best Support! - Most number of revives:", max(main_df.revives))

# 10) Max ride distance
print("The F1 Champ! - Max ride distance:", max(main_df.rideDistance), "m")

# 11) Max road kills
print("The transformer! - Most kills by a vehicle:", max(main_df.roadKills))

# 12) Max swim distance
print("The olympic champ! - Max swim distance:", max(main_df.swimDistance), "m")

# 13) Most team kills
print("The betrayer! - # of times killed a teammate:", max(main_df.teamKills))

# 14) Max vehicle destroyed
print("Vehicle Destroyer! - Max vehicles destroyed:", max(main_df.vehicleDestroys))

# 15) Max walking distance
print("The marathon champ! - Max walk distance:", max(main_df.walkDistance), "m")

# 16) Max weapons acquired
print("The collector! - Max number of weapons acquired:", max(main_df.weaponsAcquired))
