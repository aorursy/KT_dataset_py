import matplotlib.pyplot as plt

import pandas as pd



season0708 = pd.read_csv('../input/english-premier-league-tables/epl20072008.csv')

season0809 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20082009.csv')

season0910 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20092010.csv')

season1011 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20102011.csv')

season1112 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20112012.csv')

season1213 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20122013.csv')

season1314 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20132014.csv')

season1415 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20142015.csv')

season1516 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20152016.csv')

season1617 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20162017.csv')

season1718 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20172018.csv')

season1819 = pd.read_csv('../input/english-premier-league-tables-with-season/epl20182019.csv')





all_data = pd.concat([season0708, season0809, season0910, season1011, season1112, season1213, season1314, season1415, season1516, season1617, season1718, season1819], ignore_index=True)

all_data.rename(columns={"D.1": "GD"}, inplace=True)

print(all_data.head(30))
top_4_teams = (all_data.loc[all_data['#'] < 5])



print(top_4_teams.iloc[:, 0:2].groupby('Team').count())

print('\n')

print('Only ' + str(all_data['Team'].nunique()) + ' teams have competed in the Premier League over the past 12 seasons')

print('Therefore only ' + "%.2f%%" % (7/38 * 100) + ' of the teams that have competed have made the top 4.')
seasons = ['2007/08', '2008/09', '2009/10', '2010/11', '2011/12', '2012/13', '2013/14', '2014/15', '2015/16', '2016/17', '2017/18', '2018/19']



all_data_means = all_data.groupby('#').mean()

all_data_means = all_data_means.drop(['MP'], axis=1)

all_data_described = all_data.groupby('#').describe()

all_data_described = all_data_described.drop(['MP'], axis=1)

print(all_data_means.head(5))
top_4_points_mean = all_data_means.iloc[3,5]



print(sum((all_data['P'] > top_4_points_mean) & (all_data['#'] > 4)))
print(all_data.loc[(all_data["P"] > top_4_points_mean) & (all_data["#"] > 4)])
print(all_data[all_data['season'] == '2018_19'].head())
goals_scored_mean = all_data_means.iloc[0,3]



all_2nd_place = all_data[all_data['#'] == 2].reset_index()

goals_scored_mean_1st = [goals_scored_mean] * 12



plt.figure(figsize=(15,10))

ax1 = plt.subplot(2, 2, 1)

plt.plot(all_data_means['P'], color='red')

ax1.set_title('Average Points total by final position')

ax1.set_xticks(range(1,21))

ax1.set_xticklabels(range(1,21))

plt.ylabel('Total Points')

plt.xlabel('Final position')

plt.grid(True)



ax2 = plt.subplot(2, 2, 2)

plt.plot(all_data_means['F'])

ax2.set_title('Average goals scored by final position')

ax2.set_xticks(range(1,21))

ax2.set_xticklabels(range(1,21))

plt.ylabel('Total Goals scored')

plt.xlabel('Final position')

plt.grid(True)



ax3 = plt.subplot(2, 2, 3)

plt.plot(all_data_means['A'], color='green')

ax3.set_title('Average goals conceded by final position')

ax3.set_xticks(range(1,21))

ax3.set_xticklabels(range(1,21))

plt.ylabel('Total Goals conceded')

plt.xlabel('Final position')

plt.grid(True)



ax4 = plt.subplot(2, 2, 4)

plt.plot(goals_scored_mean_1st, ls='dashed')

plt.plot(all_2nd_place['F'])

ax4.set_xticks(range(0, 12))

ax4.set_xticklabels(seasons, rotation=45)

ax4.legend(['Mean goals scored 1st place team', 'Goals scored 2nd place team'],loc="upper left", prop={'size': 8})

plt.grid(True)

plt.subplots_adjust(bottom=0.15, wspace=0.5, hspace=0.5)

plt.show()
print(all_data[all_data['season'] == '2018_19'].head(10))
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



x = all_data[['F', 'A']]



y = all_data[['#']]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state=6)

mlr = LinearRegression()

model=mlr.fit(x_train, y_train)

y_predict = mlr.predict(x_test)



united_team_prediction = [[65, 39]]

wolves_team_prediction = [[72, 39]]

everton_team_prediction = [[74, 39]]

predict_united = mlr.predict(united_team_prediction)

predict_wolves = mlr.predict(wolves_team_prediction)

predict_everton = mlr.predict(everton_team_prediction)



print("Manchester United predicted finishing position: " '%.1f' % predict_united)

print("Wolves predicted finishing position: " '%.1f' % predict_wolves)

print("Everton predicted finishing position: " '%.1f' % predict_everton)
print(model.score(x_train,y_train))

print(model.score(x_test, y_test))

print(model.coef_)
plt.scatter(all_data[['F']], all_data[['#']], alpha=0.4)

plt.show()