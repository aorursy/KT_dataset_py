import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
train.head()
#load data for squad-tpp (squad) & squad-fpp
squad_data = train.loc[(train['matchType'] == 'squad') 
                       | (train['matchType'] == 'squad-fpp')]
squad_data.head()
#getting rid of useless info
useless_columns = ['Id', 'groupId', 'matchId']
squad_data.drop(useless_columns, inplace=True, axis=1)
squad_data.head()

#Average runs in a game
print("In a squad game, the whole squad or the alive players walk/sprint {:.1f}m on an average  ".format(squad_data['walkDistance'].mean()))
sns.jointplot(x="winPlacePerc", y="walkDistance",  data=squad_data, height=10, ratio=3, color="m")
plt.show()
squad_data_run = squad_data.loc[(squad_data['walkDistance'] > 1237.8 )]
squad_data_run.head()
sns.jointplot(x="winPlacePerc", y="walkDistance",  data=squad_data_run, height=10, ratio=3, color="m")
plt.show()
squad_data_run['walkDistance'].corr(squad_data_run['winPlacePerc'])
#Average damage in a game
print("In a squad game, the whole squad or the alive players deal a damage of {:.1f} on an average  ".format(squad_data['damageDealt'].mean()))
squad_runKillers = squad_data_run.loc[(squad_data_run['damageDealt'] > 132.0 )]
squad_runKillers.head()
sns.jointplot(x="winPlacePerc", y="walkDistance",  data=squad_runKillers, height=10, ratio=3, color="m")
plt.show()
squad_runKillers['walkDistance'].corr(squad_runKillers['winPlacePerc'])
squad_runCampers = squad_data_run.loc[(squad_data_run['damageDealt'] < 132.0 )]
squad_runCampers.head()
sns.jointplot(x="winPlacePerc", y="walkDistance",  data=squad_runCampers, height=10, ratio=3, color="m")
plt.show()
squad_runCampers['walkDistance'].corr(squad_runCampers['winPlacePerc'])
#Average vehicle distance traversed in a game
print("In a squad game, the whole squad or the alive players ride {:.1f}m on an average  ".format(squad_data['rideDistance'].mean()))
sns.jointplot(x="winPlacePerc", y="rideDistance",  data=squad_data_run, height=10, ratio=3, color="m")
plt.show()
squad_data_drive = squad_data.loc[(squad_data['rideDistance'] >  636.4 )]
squad_data_run.head()
sns.jointplot(x="winPlacePerc", y="rideDistance",  data=squad_data_drive, height=10, ratio=3, color="m")
plt.show()
squad_data_drive['rideDistance'].corr(squad_data_drive['winPlacePerc'])
#Average distance swam in a game
print("In a squad game, the whole squad or the alive players swim {:.1f}m on an average  ".format(squad_data['swimDistance'].mean()))
sns.jointplot(x="winPlacePerc", y="swimDistance",  data=squad_data, height=10, ratio=3, color="m")
plt.show()
squad_data_swim = squad_data.loc[(train['rideDistance'] >  4.4 )]
squad_data_run.head()
sns.jointplot(x="winPlacePerc", y="swimDistance",  data=squad_data_swim, height=10, ratio=3, color="m")
plt.show()
squad_data_swim['swimDistance'].corr(squad_data_swim['winPlacePerc'])
#Average kills in a game
print("In a squad game, the whole squad or the alive players kill {:.1f}players on an average  ".format(squad_data['kills'].mean()))
squad_camperKillers = squad_data.loc[(squad_data['walkDistance'] <  636.4 )]
squad_camperKillers['killsCategories'] = pd.cut(squad_camperKillers['kills'], [-1, 0, 2, 5, 10, 60], labels=['0_kills','1-2_kills', '3-5_kills', '6-10_kills', '10+_kills'])

plt.figure(figsize=(15,8))
sns.boxplot(x="killsCategories", y="winPlacePerc", data=squad_camperKillers)
plt.show()
squad_camperKillers['kills'].corr(squad_camperKillers['winPlacePerc'])
