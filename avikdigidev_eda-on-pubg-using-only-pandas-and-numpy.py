import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


%matplotlib inline

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#PUBG_avik_v1
#Level: Beginner
#Although I have loaded multiple libraries, I will be using only pandas and numpy in this version of kernel.

pubg_main_df = pd.read_csv('../input/train_V2.csv')

#Getting the feel of data
pubg_main_df.info()
pubg_main_df.head(20)
#Taking backup of the main data (Accidents do happen!)
pubg_exp = pubg_main_df
#If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”. 
#Value of -1 takes place of “None”.
#If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”. 
pubg_exp.loc[(pubg_exp['rankPoints'] != -1) & (pubg_exp['killPoints'] == 0), 'killPoints'] = None
pubg_exp.loc[(pubg_exp['rankPoints'] != -1) & (pubg_exp['winPoints'] == 0), 'winPoints'] = None
pubg_exp.loc[(pubg_exp['rankPoints'] == -1), 'rankPoints'] = None

pubg_exp = pubg_exp.dropna()
#Lets see what have we done
pubg_exp.info()
pubg_exp.head(20)
#Found out that now rankPoints has only one value, which is 0,
#So dropping it
pubg_exp = pubg_exp.drop(['rankPoints'], axis=1)
print("One true marksman has killed {} with headshot, had a kill count of {},\nand the most kills ever recorded(as per data) is {}.".format(pubg_exp['headshotKills'].max(), pubg_exp.set_index('kills')['headshotKills'].idxmax(), pubg_exp['kills'].max()))
print("Average person swims {:.2f} meters, travels by vehicle {:.2f} meters,\nand travels by steps {:.2f} meters.".format(pubg_exp['swimDistance'].mean(), pubg_exp['rideDistance'].mean(), pubg_exp['walkDistance'].mean()))
print("Best swimmer has swimmed {:.2f} meters.\nThe person who loved to drive the most has driven {:.2f} meters,\nThe person who preferred to travel by steps has walk/run {:.2f} meters.".format(pubg_exp['swimDistance'].max(), pubg_exp['rideDistance'].max(), pubg_exp['walkDistance'].max()))
# Variables which are going to be very useful
pubg_exp_winner = pubg_exp["winPlacePerc"] == 1
mean_damaged_winner = pubg_exp[pubg_exp_winner]["damageDealt"].mean()
mean_DBNOs_winner = pubg_exp[pubg_exp_winner]["DBNOs"].mean()
mean_kills_winner = pubg_exp[pubg_exp_winner]["kills"].mean()
mean_killStreaks_winner = pubg_exp[pubg_exp_winner]["killStreaks"].mean()
max_boosts_winner = pubg_exp[pubg_exp_winner]["boosts"].max()
mean_boosts_winner = pubg_exp[pubg_exp_winner]["boosts"].mean()
max_heals_winner = pubg_exp[pubg_exp_winner]["heals"].max()
mean_heals_winner = pubg_exp[pubg_exp_winner]["heals"].mean()
max_weaponsAcquired_winner = pubg_exp[pubg_exp_winner]["weaponsAcquired"].max()
mean_weaponsAcquired_winner = pubg_exp[pubg_exp_winner]["weaponsAcquired"].mean()
max_vehicleDestroys = pubg_exp["vehicleDestroys"].max()
print("The person who ends up winning the match usually deals {:.2f} damage or less, knocks down at least {:.1f} enemies \nhad a kill count of at least {:.1f} enemies , and the best player had a killstreak of {}. ONLY!??".format(mean_damaged_winner, mean_DBNOs_winner, round(mean_kills_winner), round(mean_killStreaks_winner)))
print("".format())
print("{} Fortuitous noobs won without killing anyone at all.".format(len(pubg_exp[(pubg_exp['kills'] == pubg_exp['kills'].min()) & (pubg_exp['winPlacePerc'] == pubg_exp['winPlacePerc'].max())])))
print("{} Confident Players had their chicken dinner without using any boosts".format(len(pubg_exp[(pubg_exp['boosts'] == pubg_exp['boosts'].min()) & (pubg_exp['winPlacePerc'] == pubg_exp['winPlacePerc'].max())])))
print("{} Iron-heart players won without using any bandages, medkit or first-aid.".format(len(pubg_exp[(pubg_exp['heals'] == pubg_exp['heals'].min()) & (pubg_exp['winPlacePerc'] == pubg_exp['winPlacePerc'].max())])))
print("{} Marksman Player won with killing most with the {} headshots, while at an average a winner kills {} enemies with headshots(are we that bad!?)".format((len(pubg_exp[(pubg_exp['headshotKills'] == pubg_exp['headshotKills'].max()) & (pubg_exp['winPlacePerc'] == pubg_exp['winPlacePerc'].max())])),pubg_exp['headshotKills'].max(),(len(pubg_exp[(pubg_exp['headshotKills'] == pubg_exp['headshotKills'].mean()) & (pubg_exp['winPlacePerc'] == pubg_exp['winPlacePerc'].max())]))))
print("{} Hyperactive player was most anxious and won using most boosts (Keep calm and have Chicken Dinner), he used {} boosts, while an average winner uses at least {} boosts.".format(len(pubg_exp[(pubg_exp['boosts'] == max_boosts_winner) & (pubg_exp['winPlacePerc'] == 1)]), max_boosts_winner, round(mean_boosts_winner)))
print("{} That hard-to-kill player won using most of healing items like bandages, medkit or first-aid, he used {} heals, while an average winner uses {} bandages/medkit/first-aids.".format(len(pubg_exp[(pubg_exp['heals'] == max_heals_winner) & (pubg_exp['winPlacePerc'] == 1)]),max_heals_winner, round(mean_heals_winner)))

print("{} player who won switching weapons the most, he used {} weapons, while an average winner switches {} weapons.".format(len(pubg_exp[(pubg_exp['weaponsAcquired'] == max_weaponsAcquired_winner) & (pubg_exp['winPlacePerc'] == 1)]),max_weaponsAcquired_winner, round(mean_weaponsAcquired_winner)))
print("{} is the max count of vehciles which were destroyed in a match.".format(max_vehicleDestroys))
print("{} people couldn't kill anyone during the match.".format(len(pubg_exp[(pubg_exp['kills'] == pubg_exp['kills'].min())])))
count_dict = dict(zip(*np.unique(pubg_exp["matchType"].values, return_counts=True)))
print ( "Out of {} total matches:".format(len(pubg_exp)))
for k in count_dict.keys():
    print ("{} matches played of type {} ".format(count_dict[k], k))
for k in count_dict.keys():
    print("From the given data we can also see that {}, {} were winners.".format(k, len(pubg_exp[(pubg_exp['matchType'] == k) & (pubg_exp['winPlacePerc'] == pubg_exp['winPlacePerc'].max())])))
#You reached this point! This was my first kernel.
#An upvote would be appreciated and help me keep going!
#Please suggest if it can be improved or I made any mistake:)
#About using other libraries, WORK IN PROGRESS
