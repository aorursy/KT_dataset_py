# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib import rcParams

# reading file from data source

file_path = "../input/international-football-results-from-1872-to-2017/results.csv"

data = pd.read_csv(file_path)

# added an extra column named year to the dataset for further analytics 

data["year"] = data['date'].str[:4].astype(int)

# printing the first five rows or observations

data.head()

# functions for determining win,lose and draw for both home and away performances

def home_performance(s):

    if (s.home_score > s.away_score):

        return "Win"

    elif (s.home_score < s.away_score):

        return "Lose"        

    else:

        return "Draw"

def away_performance(s):

    if (s.away_score > s.home_score):

        return "Win"

    elif (s.away_score < s.home_score):

        return "Lose"        

    else:

        return "Draw"

    

# adding two columns to dataset named home and away performance for getting the result

data['home_performance'] = data.apply(home_performance,axis=1)

data['away_performance'] = data.apply(away_performance,axis=1)



# seggregating argentina and brazil data

argentina_stat = data[(data.home_team == 'Argentina') | (data.away_team == 'Argentina')]

brazil_stat = data[(data.home_team == 'Brazil') | (data.away_team == 'Brazil')]



# getting count for win,lose and draw for both home and away

home_perf_arg = argentina_stat[argentina_stat.home_team == 'Argentina'].home_performance.value_counts()

away_perf_arg = argentina_stat[argentina_stat.away_team == 'Argentina'].away_performance.value_counts()

home_perf_brz = brazil_stat[brazil_stat.home_team == 'Brazil'].home_performance.value_counts()

away_perf_brz = brazil_stat[brazil_stat.away_team == 'Brazil'].away_performance.value_counts()



# creating a figure object to plot the bar graph. We have created four axes to show four bar graph for both the team

fig, ax = plt.subplots(2,2, figsize=(8,4), sharey=True, dpi=120)

plt.subplots_adjust(top=1.5,bottom = .1)



# setting up the title for the axes.

ax[0,0].set_title("Argentina Home Performance: " + str(home_perf_arg.values.sum()) + " games", fontsize = 10)

ax[0,1].set_title("Argentina Away Performance: " + str(away_perf_arg.values.sum()) + " games", fontsize = 10)

ax[1,0].set_title("Brazil Home Performance: " + str(home_perf_brz.values.sum()) + " games", fontsize = 10)

ax[1,1].set_title("Brazil Away Performance: "  + str(away_perf_brz.values.sum()) + " games", fontsize = 10)



# setting style for the graph

plt.style.use(['seaborn-dark-palette','ggplot'])



# plotting data in seaborn plot

sns.barplot(x = home_perf_arg.keys(), y = home_perf_arg.values, ax = ax[0,0],palette=("Blues_d"))

sns.barplot(x = away_perf_arg.keys(), y = away_perf_arg.values, ax = ax[0,1],palette=("Blues_d"))

sns.barplot(x = home_perf_brz.keys(), y = home_perf_brz.values, ax = ax[1,0],palette=("Blues_d"))

sns.barplot(x = away_perf_brz.keys(), y = away_perf_brz.values, ax = ax[1,1],palette=("Blues_d"))

# this function returns a list containing performance data which looks like

# [['Friendly','100','20','30'],['World Cup','100','20','30'],['Copa','100','20','30']]

def calculate_results(team_data, team):

#     sorting out tournament category

    keys = team_data.tournament.value_counts().keys()

    performance_data = []

    for each in keys:

        a = []

        win = team_data[((team_data.tournament == each) & (((team_data.home_team == team) & (team_data.home_performance == 'Win')) | 

                           ((team_data.away_team == team) & (team_data.away_performance == 'Win'))))].home_team.count()

        lose =  team_data[((team_data.tournament == each) & (((team_data.home_team == team) & (team_data.home_performance == 'Lose')) | 

                           ((team_data.away_team == team) & (team_data.away_performance == 'Lose'))))].away_team.count()

        draw = team_data[((team_data.tournament == each) & (team_data.home_performance == "Draw"))].home_team.count()

        a.extend([each,win,lose,draw])

        performance_data.append(a)

    return performance_data

# creating dataframe with the result got from the function for both the team

df_arg = pd.DataFrame(columns = ['Tournament','Win','Lose','Draw'],data=calculate_results(argentina_stat, "Argentina"))

df_brz = pd.DataFrame(columns = ['Tournament','Win','Lose','Draw'],data=calculate_results(brazil_stat, "Brazil"))

df_arg.set_index('Tournament',inplace = True)

df_brz.set_index('Tournament',inplace = True)

# creating figure object which contains two columns in a single row for plotting both team's result

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,3), sharey=True, dpi=200)

plt.subplots_adjust(top=1.5,bottom = .05)

# setting style for the graph

plt.style.use(['seaborn-deep','dark_background','ggplot','seaborn-dark-palette'])

df_arg.plot.bar(ax = ax1)

# setting the first column title , xlabel and ylabel

ax1.set_title("Argentina Stat")

ax1.set_xlabel("Tournament")

ax1.set_ylabel("Count")

ax1.set_xticklabels(argentina_stat.tournament.value_counts().keys(),rotation=90)



df_brz.plot.bar(ax = ax2)

# setting the first column title , xlabel and ylabel

ax2.set_title("Brazil Stat")

ax2.set_xlabel("Tournament")

ax2.set_ylabel("Count")

ax2.set_xticklabels(brazil_stat.tournament.value_counts().keys(),rotation=90)

plt.show()
# this function is quite similliar to the previous one. It's also returns a list 

# but contains city information in place of tournament

def home_field_ratio(data,team_name):

    home_field_count = data[(data.home_team == team_name) & 

                                      (data.neutral == False)].city.value_counts()[:7]

    home_performance_result = []

    for each in home_field_count.keys():

        a = []

        win = data[(data.home_team == team_name) & (data.neutral == False) &

                            (data.home_performance == 'Win') & (data.city == each)].city.count()

        lose = data[(data.home_team == team_name) & (data.neutral == False) &

                            (data.home_performance == 'Lose') & (data.city == each)].city.count()

        draw = data[(data.home_team == team_name) & (data.neutral == False) &

                            (data.home_performance == 'Draw') & (data.city == each)].city.count()

        win_percentage = (win/home_field_count[each])* 100

        lose_percentage = (lose/home_field_count[each])* 100

        draw_percentage = (draw/home_field_count[each])* 100



        a.extend([each,win_percentage,lose_percentage,draw_percentage])

        home_performance_result.append(a)

    return home_performance_result



# setting up dataframe for argentina with the list obtained from the above function

df_arg_home_perf = pd.DataFrame(home_field_ratio(argentina_stat,'Argentina'),

                                columns = ["city","win","lose","draw"])

df_arg_home_perf.set_index('city',inplace = True)



# setting up dataframe for brazil with the list obtained from the above function

df_brz_home_perf = pd.DataFrame(home_field_ratio(brazil_stat,'Brazil'),

                                columns = ["city","win","lose","draw"])

df_brz_home_perf.set_index('city',inplace = True)



# creating figure object, quite similar to the previous one 

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), sharey=True, dpi=200)

plt.style.use(['seaborn-deep','dark_background','ggplot','seaborn-dark-palette'])

df_arg_home_perf.plot.bar(ax = ax1)

df_brz_home_perf.plot.bar(ax = ax2)

ax1.set_title("Home Performance Argentina")

ax2.set_title("Home Performance Brazil")
def win_percentage(data,team):    

    match_count_year = data.year.value_counts()

    foo = match_count_year[match_count_year>=10]

    final_result = []

    for each in foo.keys():

        ls = []

        win = data[((data.home_team == team) & (data.home_performance == 'Win') &

                            (data.year == each)) | ((data.away_team == team) & (data.away_performance == 'Win') &

                            (data.year == each))].year.count()

        win_percentage = (win/foo[each])*100

        ls.extend([each,win_percentage])

        final_result.append(ls)

    return final_result

win_curve_arg = pd.DataFrame(win_percentage(argentina_stat, 'Argentina'),columns=['Year','Win_Percentage'])

win_curve_brz = pd.DataFrame(win_percentage(brazil_stat, 'Brazil'),columns=['Year','Win_Percentage'])

# win_curve.set_index('Year',inplace = True)

fig, (ax1,ax2) = plt.subplots(1,2, figsize=(16,5))

sns.lineplot(x = win_curve_arg.Year, y = win_curve_arg.Win_Percentage,ax = ax1)

sns.lineplot(x = win_curve_brz.Year, y = win_curve_brz.Win_Percentage,ax = ax2)
comp = data.copy()

comp = comp[((comp.home_team == 'Argentina') & (comp.away_team == 'Brazil')) | 

            ((comp.home_team == 'Brazil') & (comp.away_team == 'Argentina'))]

comp_arg = pd.DataFrame(calculate_results(comp,'Argentina'),columns = ['Tournament','Arg_win','Brazil_win','Draw'])

comp_arg.set_index('Tournament',inplace = True)

plt.style.use(['seaborn-deep','dark_background','seaborn-dark-palette'])

comp_arg.plot.bar(figsize=(15,5),rot = 90)
