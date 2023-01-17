import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

import operator

import re

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import RandomizedSearchCV
players_df = pd.read_csv("../input/data.csv")

players_df.head()
players_df = players_df.drop(columns=['ID','Photo','Club Logo','Work Rate','Body Type','Real Face','Loaned From',

                                     'LS','ST','RS','LW','LF','CF','RF','RW','LAM','CAM','RAM','LM','LCM','CM',

                                     'RCM','RM','LWB','LDM','CDM','RDM','RWB','LB','LCB','CB','RCB','RB'])

players_df = players_df.drop(players_df.columns[0], axis=1)

players_df.info()
players_df = players_df.dropna(axis = 0, how = 'any')

players_df.info()
def findValue(value):

    splitted = re.split('(\d+\.\d+|\d+)([A-Z])',value)

    try:

        if splitted[2] is 'M':

            return pd.to_numeric(splitted[1])

        else:

            return pd.to_numeric(splitted[1])/1000

    except IndexError:

        return 0



def findWages(value):

    return re.findall('\d+\.\d+|\d+',value)[0]



def findAttribute(value):

    return re.findall('\d+',value)

            

players_df['Value'] = players_df['Value'].apply(findValue)

players_df['Wage'] = players_df['Wage'].apply(findWages)

players_df['Value'] = pd.to_numeric(players_df['Value'])

players_df['Wage'] = pd.to_numeric(players_df['Wage'])

players_df
countries = list(players_df.groupby(['Nationality']).groups.keys())

temp = players_df.groupby(['Nationality']).count()

temp = list(temp.loc[:,"Name"])

countries_dict = dict(zip(countries,temp))

sorted_countries = sorted(countries_dict.items(), key=operator.itemgetter(1),reverse=True)

countries_dict = dict(sorted_countries)

countries_dict = {k: countries_dict[k] for k in list(countries_dict)[:5]}

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

with plt.style.context(('ggplot')):

    plt.bar(countries_dict.keys(), countries_dict.values(),color="grey")

    plt.xlabel('Countries ------>')

    plt.ylabel('Number of players -------->')

    plt.title('Player-Nationality Distribution')

    plt.show()
overall = list(players_df.loc[:,"Overall"])

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

with plt.style.context(('ggplot')):

    plt.hist(overall, bins=12, ec='black', alpha=0.8, color='blue')

    plt.xlabel("Overall ------->")

    plt.ylabel("Number of players --------->")

    plt.xticks([50,55,60,65,70,75,80,85,90,95,100])

    plt.title("Overall Distribution in FIFA 19")

    plt.show()
gk_defenders = players_df.loc[players_df['Position'].isin(['GK','RWB','RB','RCB','CB','LCB','LB','LWB'])]

figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

sns.boxplot(x= gk_defenders['Position'], y=gk_defenders['Overall']).set_title('Range of Goalkeeper and Defender Overalls')

midfielders = players_df.loc[players_df['Position'].isin(['RM','CM','LM','CDM','RDM','LDM','RAM','CAM','LAM'])]

figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')

sns.boxplot(x=midfielders['Position'],y=midfielders['Overall']).set_title('Range of Midfielder Overalls')

forwards = players_df.loc[players_df['Position'].isin(['RW','CF','LW','ST','RS','LS'])]

figure(num=None, figsize=(10,6), dpi=80, facecolor='w', edgecolor='k')

sns.boxplot(x=forwards['Position'], y=forwards['Overall']).set_title('Range of Forward Overalls')
heatmap_df = players_df.filter(['Position','Crossing','Finishing','HeadingAccuracy','ShortPassing','Volleys','Dribbling',

                               'Curve','FKAccuracy','LongPassing','BallControl','Acceleration','SprintSpeed','Agility',

                                'Reactions','Balance','ShotPower','Jumping','Stamina','Strength','LongShots','Aggression',

                               'Interceptions','Positioning','Vision','Penalties','Composure','Marking','StandingTackle',

                               'SlidingTackle']).groupby('Position').mean()

heatmap_df_defenders = heatmap_df.loc[['RWB','RB','RCB','CB','LCB','LB','LWB']]

heatmap_df_defenders = heatmap_df_defenders.round().astype(int)

figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')

sns.heatmap(heatmap_df_defenders,annot=heatmap_df_defenders,fmt="",cmap="RdYlGn",linewidths=0.3)

plt.title('Relation between Abilities and Defending Positions')

plt.show()
heatmap_df_midfielders = heatmap_df.loc[['RM','CM','LM','CDM','RDM','LDM','RAM','CAM','LAM']]

heatmap_df_midfielders = heatmap_df_midfielders.round().astype(int)

figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')

sns.heatmap(heatmap_df_midfielders,annot=heatmap_df_midfielders,fmt="",cmap="RdYlGn",linewidths=0.3)

plt.title('Relation between Abilities and Midfield Positions')

plt.show()
heatmap_df_attackers = heatmap_df.loc[['RW','CF','LW','ST','RS','LS']]

heatmap_df_attackers = heatmap_df_attackers.round().astype(int)

figure(num=None, figsize=(15, 6), dpi=80, facecolor='w', edgecolor='k')

sns.heatmap(heatmap_df_attackers,annot=heatmap_df_attackers,fmt="",cmap="RdYlGn",linewidths=0.3)

plt.title('Relation between Abilities and Attacking Positions')

plt.show()
overall = players_df.groupby(['Age']).mean()

overall_list = list(overall.loc[:,"Overall"])

potential_list = list(overall.loc[:,"Potential"])

age = list(players_df.loc[:,"Age"].unique())

age = sorted(age)

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

with plt.style.context(('ggplot')):

    plt.xticks([17,22,27,32,37,42])

    plt.plot(age,overall_list,color='orange', linewidth=4, alpha=0.7, label='Overall')

    plt.plot(age,potential_list, color='grey', linewidth=1, alpha=0.4, label='Potential')

    plt.legend(loc='upper right')

    plt.xlabel("Age -------->")

    plt.ylabel("Overall/Potential ---------->")

    plt.title("Overall and Potential Distribution in FIFA 19 with respect to Age")

    plt.grid(b=None, which='major', axis='both', color='w')

    plt.show()
valuation = players_df.groupby(['Age']).mean()

valuation_list = list(valuation.loc[:,'Value'])

age = list(players_df.loc[:,'Age'].unique())

age = sorted(age)

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

with plt.style.context('ggplot'):

    plt.plot(age,valuation_list)

    plt.title("Variation of Player Value with Age")

    plt.xlabel("Age --------->")

    plt.ylabel("Valuation in millions --------->")

    plt.show()
wages = players_df.groupby(['Age']).mean()

wages_list = list(valuation.loc[:,'Wage'])

age = list(players_df.loc[:,'Age'].unique())

age = sorted(age)

figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

with plt.style.context('ggplot'):

    plt.plot(age,wages_list)

    plt.title("Variation of Player Wages with Age")

    plt.xlabel("Age --------->")

    plt.ylabel("Wages in thousands --------->")

    plt.show()
value_list = list(players_df.loc[:,'Value'])

overall = list(players_df.loc[:,'Overall'])

figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

with plt.style.context('ggplot'):

    plt.scatter(overall,value_list,s=15,alpha=0.8,c='grey')

    plt.xlabel('Overall -------->')

    plt.ylabel('Valuation in millions -------->')

    plt.title('Variation of Valuation with Overall')

    plt.show()
clubs_df = players_df.groupby('Club').mean()

clubs_df_youngest = clubs_df.sort_values(['Age'],ascending=True)

clubs_df_youngest = clubs_df_youngest['Age']

clubs_df_youngest = clubs_df_youngest[0:10]

clubs_df_eldest = clubs_df.sort_values(['Age'],ascending=False)

clubs_df_eldest = clubs_df_eldest['Age']

clubs_df_eldest = clubs_df_eldest[0:10]

clubs_youngest_eldest = pd.concat([clubs_df_youngest, clubs_df_eldest])

clubs_youngest_plot = list(clubs_df_youngest)

clubs_youngest_plot[:] = [age - 15 for age in clubs_youngest_plot]

clubs_eldest_plot = list(clubs_df_eldest)

clubs_eldest_plot[:] = [age - 25 for age in clubs_eldest_plot]

clubs_eldest_plot[:] = [-age for age in clubs_eldest_plot]

clubs_plot = clubs_youngest_plot + clubs_eldest_plot

youngest_eldest_clubs = list(clubs_youngest_eldest.index.values)

with plt.style.context('ggplot'):

    figure(num=None, figsize=(15, 8), dpi=80, edgecolor='k')

    plt.barh(youngest_eldest_clubs, clubs_plot)

    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    clubs_youngest_eldest = list(clubs_youngest_eldest)

    plt.show()
talents_df = players_df.filter(["Name", "Club", "Overall"])

talents_df = talents_df[talents_df.Overall >= 85]

talents_df = talents_df.groupby("Club").count()

talents_df = talents_df.sort_values(by=['Name'],ascending=False)

talents_df = talents_df[:10]

talents_clubs = list(talents_df.index.values)

talents_count = list(talents_df.iloc[:, 1])

with plt.style.context('ggplot'):

    figure(num=None, figsize=(15, 8), dpi=80, edgecolor='k')

    plt.barh(talents_clubs, talents_count)

    plt.title('Top 10 talented clubs in FIFA 19')

    plt.xlabel('Count ---------->')

    plt.show()
talented_df = players_df.filter(['Name', 'Club', 'Overall'])

talented_df = talented_df.groupby('Club').mean()

talented_df = talented_df.sort_values(by=['Overall'], ascending=False)

talented_df = talented_df[:10]

talented_clubs = talented_df.index.values

players_talented_clubs = players_df.filter(['Name','Club','Overall'])

players_talented_clubs = players_talented_clubs.loc[players_talented_clubs['Club'].isin(talented_clubs)]

players_talented_clubs

with plt.style.context('ggplot'):

    figure(num=None, figsize=(14, 6), dpi=80, facecolor='w', edgecolor='k')

    sns.violinplot(x= players_talented_clubs['Club'], y=players_talented_clubs['Overall']).set_title('Overall distribution of the top 10 talented clubs in FIFA 19')
potential_df = players_df.filter(['Name','Age','Overall','Potential'])

potential_df['Growth'] = potential_df['Potential'] - potential_df['Overall']

potential_df = potential_df.sort_values(by=['Growth'],ascending=False)

potential_df = potential_df.iloc[:10]

players_list = list(potential_df.iloc[:, 0])

overall_list = list(potential_df.iloc[:,2])

growth_list = list(potential_df.iloc[:,4])

with plt.style.context('ggplot'):

    figure(num=None, figsize=(15, 8), dpi=80, edgecolor='k')

    plt.barh(players_list, overall_list, color='green', label='Overall')

    plt.barh(players_list, growth_list, left=overall_list, color='green', label='Growth', alpha=0.3)

    plt.legend()

    plt.title('Top 10 Players with the best growth potential')

    plt.xlabel('Overall -------------->')

    plt.show()
valuation_df = players_df.filter(['Name','Club','Value','Wage'])

valuation_df = valuation_df.groupby('Club').sum()

valuation_df = valuation_df.sort_values(by=['Value'], ascending=False)

valuation_df = valuation_df[:10]

clubs_list = list(valuation_df.index.values)

value_list = list(valuation_df.iloc[:, 0])

wage_list = list(valuation_df.iloc[:, 1])

alpha_list = wage_list

wage_list[:] = [wages / 5 for wages in wage_list]

#alpha_list[:] = [alpha/1004 for alpha in alpha_list]

rgba_colors = np.zeros((10,4))

rgba_colors[:,0] = 1.0

rgba_colors[:, 3] = alpha_list

with plt.style.context('ggplot'):

    figure(num=None, figsize=(15, 8), dpi=80, edgecolor='k')

    plt.scatter(value_list, clubs_list, s = wage_list )

    plt.show()
pot = list(players_df.iloc[:,5])

val = list(players_df.iloc[:,7])

figure(num=None, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')

with plt.style.context('ggplot'):

    plt.scatter(pot,val,s=15,alpha=0.8,c='grey')

    plt.xlabel('Potential -------->')

    plt.ylabel('Valuation in millions -------->')

    plt.title('Variation of Valuation with Potential')

    plt.show()
X = players_df.iloc[:,4:6]

y = players_df.iloc[:,7]

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.20, random_state = 42)

print('Training matrix of features shape: ', train_X.shape)

print('Training dependent variable shape: ', train_y.shape)

print('Test matrix of features shape: ', test_X.shape)

print('Test dependent variable shape: ', test_y.shape)
regressor = RandomForestRegressor(n_estimators=10, random_state=42)

regressor.fit(train_X, train_y)

predictions = regressor.predict(test_X)

errors = abs(predictions - test_y)

print('Mean absolute error: ',round(np.mean(errors),2) )

#Mean absolute Percentage error

mape = 100 * (errors/test_y)



#Calculating accuracy

acc = 100 - np.mean(mape)

print('Accuracy: ', round(acc,2), ' %')
print(regressor.get_params())
n_estimators = [int(x) for x in np.linspace(start=200, stop= 2000, num=10)]

max_features = ['auto','sqrt']

max_depth = [int(x) for x in np.linspace(10,110,num=11)]

max_depth.append(None)

min_samples_split = [2,5,10]

min_samples_leaf = [1,2,4]

bootstrap = [True,False]

random_grid = {'n_estimators': n_estimators,

              'max_features': max_features,

              'max_depth': max_depth,

              'min_samples_split': min_samples_split,

              'min_samples_leaf': min_samples_leaf,

              'bootstrap': bootstrap}

regressor_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid, 

                                     n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)

regressor_random.fit(train_X, train_y)
regressor_random.best_params_
predictions_random = regressor_random.predict(test_X)

errors = abs(predictions_random - test_y)

print('Mean absolute error: ',round(np.mean(errors),2) )

#Mean absolute Percentage error

mape = 100 * (errors/test_y)



#Calculating accuracy

acc = 100 - np.mean(mape)

print('Accuracy: ', round(acc,2), ' %')