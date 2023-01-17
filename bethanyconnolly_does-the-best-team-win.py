import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np  

from scipy import stats  
from operator import itemgetter 
from pathlib import Path

# Load the datafiles
data_directory = Path('../input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/')
regular_season = pd.read_csv(data_directory / "MRegularSeasonCompactResults.csv")
tourney_results = pd.read_csv(data_directory / "MNCAATourneyCompactResults.csv")
tourney_seeds = pd.read_csv(data_directory / "MNCAATourneySeeds.csv")

# Drop the year 2020 (cancelled)
regular_season = regular_season.drop(regular_season[regular_season.Season == 2020].index)

# Get the unique team IDs for each competition year
tourney_teams = tourney_seeds.groupby('Season').TeamID.unique()

# Make a dictionary of years containing how many times each team won a game in the regular season
year_team_dict = {}
for year, year_group in regular_season.groupby('Season'):
    year_team_dict[year] = {}
    for win_team, win_team_group in year_group.groupby('WTeamID'):
        if win_team in tourney_teams[year]:
            year_team_dict[year][win_team] = win_team_group.DayNum.values

# Plot the results of the cumulative win count for each team in the 2019 season
%config InlineBackend.figure_format = 'svg'
plt.figure(figsize = (10, 5))
for team, days in year_team_dict[2019].items():
    plt.plot(days, range(len(days)))
    plt.xlabel("Regular Season (Days)", fontsize = 12)
    plt.ylabel("Cumulative Wins", fontsize = 12)
    plt.title("Cumulative Wins Per Team Over Regular Season: 2019", fontsize = 14, fontweight = 'bold')  
plt.show()
# Obtain the final number of matches won per team, per year
year_team_cumulative = {}
for year in year_team_dict:
    year_team_cumulative[year] = {}
    for team, days in year_team_dict[year].items():
        year_team_cumulative[year][team] = len(days) 


# Figure of histograms for each year 1985 - 2019        
plt.figure(figsize = (14, 16))
for num, year in enumerate(year_team_cumulative):

    # Color map for histogram
    cm = plt.cm.get_cmap('plasma')
    ytc = list(year_team_cumulative[year].values())
    
    # Make the hisogram
    plt.subplot(7,5,(num + 1))
    n, bins, patches = plt.hist(ytc, density = True, alpha = 0.9, edgecolor='black')
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Set the bin color, based on bin value
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
        
    # Labels and title
    plt.xlabel("Regular Season Win Count", fontsize = 9)
    plt.ylabel("Frequency Normalized", fontsize = 9)
    plt.title(year, fontsize = 12, fontweight = 'bold')
    
    # Set the xticks 
    xt = plt.xticks()[0]  
    xmin, xmax = min(xt), max(xt)  
    lnspc = np.linspace(xmin, xmax, len(ytc))
    
    # Fit the normal curve
    m, s = stats.norm.fit(ytc)   
    pdf_g = stats.norm.pdf(lnspc, m, s) 
    plt.plot(lnspc, pdf_g, color = 'black')

    # Show top 25th percent
    q3 = np.percentile(ytc, 75)
    maxq = np.percentile(ytc, 100)
    plt.axvspan(q3, maxq, color = 'blue', alpha = 0.2)
   
    plt.subplots_adjust(hspace = 1, wspace=0.45)   
    
plt.show()
# Get a dictionary of top 16 teams from the preseason based on win count
regular_topteams = {}
n = 16
for year in year_team_cumulative:
    top_n = dict(sorted(year_team_cumulative[year].items(), key = itemgetter(1), reverse = True)[:n])
    regular_topteams[year] = {}
    for team, score in year_team_cumulative[year].items():
        regular_topteams[year] = list(top_n.keys()) 

# Get a dictionary of 'sweet 16' teams each year based on teams which win on days 138/139
tournament_top_teams = {}
for year, year_group in tourney_results.groupby('Season'):
    top_teams = year_group[(year_group['DayNum'] == 138) | (year_group['DayNum'] == 139)] 
    tournament_top_teams[year] = list(top_teams.WTeamID)
    
# Calculate accuracy score of regular season top 16 predictions from tournament matches
score = {}
for year in tournament_top_teams:
    correct_teams = []
    for team in tournament_top_teams[year]:
        if team in regular_topteams[year]:
            correct_teams.append(team)
    team_predict_score = len(correct_teams)/ n    
    score[year] = team_predict_score   
scores = list(score.values())

# Calcuate average prediction across all years
mean_score = sum(scores) / len(scores)

# Calculate mean error in the accuracy score
std = np.std(scores)

# Baseline calculated as random chance that any 16 teams make it to the semi-final
baseline = (16/68)

# Scatterplot of accuracy each year
plt.figure(figsize = (10, 5))
ax = plt.subplot()
colors = scores
plt.scatter(score.keys(), score.values(), cmap = 'plasma', c = colors, edgecolors = 'black', marker='o', s = 75)
plt.title("Semi-Finalists Prediction Accuracy: Win Count", fontsize = 14, fontweight = 'bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Win Count Prediction Accuracy", fontsize = 12)
plt.hlines(mean_score, xmin = 1985, xmax = 2019, color = 'red', label = 'Mean Prediction Accuracy', linestyles='dashed', alpha = 0.7)
plt.text(2009,0.95,'- Mean Prediction Accuracy',rotation=0, color = 'red', alpha = 0.8, fontsize = 11)
plt.text(2009,0.9,'- Random Prediction Accuracy',rotation=0, color = 'blue', alpha = 0.8, fontsize = 11)
plt.hlines(baseline, xmin = 1985, xmax = 2019, color = 'blue', label = 'Random Prediction Accuracy', linestyles='dashed', alpha = 0.7)
ax.set_ylim(ymin=0, ymax = 1)
plt.show()
# List of scores (win and loose games) for each team in the regular season
year_team_dict2 = {}
for year, year_group in regular_season.groupby('Season'):
    year_team_dict2[year] = {}
    for team, team_group in year_group.groupby('WTeamID'):
        if team in tourney_teams[year]:
            year_team_dict2[year][team] = list(team_group.WScore.values)
    for team, team_group in year_group.groupby('LTeamID'):
        if team in tourney_teams[year]:
            year_team_dict2[year][team].extend(team_group.LScore)

# Calculate each teams cumulative win score each year 
year_team_cumulative_score = {}
for year in year_team_dict2:
    year_team_cumulative_score[year] = {}
    for team, score in year_team_dict2[year].items():
        year_team_cumulative_score[year][team] = sum(score)
        
# Figure of histograms for each year 1985 - 2019   
plt.figure(figsize = (14, 16))
for num, year in enumerate(year_team_cumulative):
     
    # Color map for histogram
    cm = plt.cm.get_cmap('plasma') #color map is plasma
    ytc = list(year_team_cumulative_score[year].values()) # list of scores
    
    # make the hisogram
    plt.subplot(7,5,(num+1))
    n, bins, patches = plt.hist(ytc, density = True, alpha = 0.9, edgecolor='black')#plot each histogram
    
    # Set the bin color, based on bin value
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    col = bin_centers - min(bin_centers)
    col /= max(col)
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
     
    # Labels and title
    plt.xlabel("Cumulative Score", fontsize = 9)
    plt.ylabel("Frequency Normalized", fontsize = 9)
    plt.title(year, fontsize = 12, fontweight = 'bold')
    
    # Set the xticks 
    #xmin, xmax = 0, 40 
    xt = plt.xticks()[0]  
    xmin, xmax = min(xt), max(xt)  
    lnspc = np.linspace(xmin, xmax, len(ytc))
    
    # Fit the normal curve.
    m, s = stats.norm.fit(ytc)   
    pdf_g = stats.norm.pdf(lnspc, m, s) 
    plt.plot(lnspc, pdf_g, color = 'black')

    # Show top 25th percent
    q3 = np.percentile(ytc, 75)
    maxq = np.percentile(ytc, 100)
    plt.axvspan(q3, maxq, color = 'blue', alpha = 0.2)
   
    plt.subplots_adjust(hspace = 1, wspace=0.6)   
    
plt.show()
# Dictionary of top 16 teams in the regular season based on cumulative score
regular_topteams_score = {}
n = 16
for year in year_team_cumulative_score:
    top_n = dict(sorted(year_team_cumulative_score[year].items(), key = itemgetter(1), reverse = True)[:n])
    regular_topteams_score[year] = {}
    for team, score in year_team_cumulative_score[year].items():
        regular_topteams_score[year] = list(top_n.keys()) 

# Teams in the tournament semi_final 
tournament_top_teams_score = {}
for year, year_group in tourney_results.groupby('Season'):
    top_teams = year_group[(year_group['DayNum'] == 138) | (year_group['DayNum'] == 139)] 
    tournament_top_teams_score[year] = list(top_teams.WTeamID)

# Calculate accuracy score of cumulative score top 16 predictions from tournament matches
for year in tournament_top_teams_score:
    correct_teams = []
    for team in tournament_top_teams_score[year]:
        if team in regular_topteams_score[year]:
            correct_teams.append(team)
    team_predict_score = len(correct_teams)/16
    score2[year] = team_predict_score   

# Calcuate average prediction across all years
scores2 = list(score2.values())
mean_score2 = sum(scores2) / len(scores2)

# Calculate mean error in the accuracy score
std2 = np.std(scores)

# Baseline calculated as random chance that any 16 teams make it to the semi-final
baseline2 = (16/68)

# Scatterplot of cumulative score prediction accuracy each year
plt.figure(figsize = (10, 5))
ax = plt.subplot()
colors = scores2
plt.scatter(score2.keys(), score2.values(), cmap = 'plasma', c = colors, edgecolors = 'black', marker='o', s = 75)
plt.title("Semi-Finalists Prediction Accuracy: Cumulative Score", fontsize = 14, fontweight = 'bold')
plt.xlabel("Year", fontsize=12)
plt.ylabel("Total Score Prediction Accuracy", fontsize=12)
plt.hlines(mean_score2, xmin = 1985, xmax = 2019, color = 'red', label = 'Mean Prediction Accuracy', linestyles='dashed', alpha = 0.7)
plt.text(2009,0.95,'- Mean Prediction Accuracy',rotation=0, color = 'red', alpha = 0.9, fontsize = 11)
plt.text(2009,0.9,'- Random Prediction Accuracy',rotation=0, color = 'blue', alpha = 0.9, fontsize = 11)
plt.hlines(baseline2, xmin = 1985, xmax = 2019, color = 'blue', label = 'Random Prediction Accuracy', linestyles='dashed', alpha = 0.7)
ax.set_ylim(ymin=0, ymax = 1)

plt.show()

# Histograms figure comparing frequency of prediction accuracy for each metric
fig = plt.subplots(1,2, figsize = (14, 6))

# Histogram for Semi-Finalists Prediction Frequency (Win Count)
cm = plt.cm.get_cmap('plasma')
ax = plt.subplot(1,2,1)
n, bins, patches = plt.hist(scores, bins = 5, density = True, edgecolor='black', alpha = 0.9)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))
plt.xlabel("Prediction Accuracy", fontsize = 12)
plt.ylabel("Frequency", fontsize = 12)
plt.title("Semi-Finalists Prediction Frequency: Win Count", fontsize = 13, fontweight = 'bold')
ax.set_xlim(xmin=0, xmax = 1)
plt.axvline(mean_score, color = 'black', label = 'Mean Prediction Accuracy', linestyle='dashed')
plt.text(0.62,3.85,'Mean Prediction Accuracy',rotation=0, color = 'black', fontsize = 9.5)
plt.text(0.62,3.68,'Random Prediction Accuracy',rotation=0, color = 'blue', fontsize = 9.5)
plt.axvline(baseline, color = 'blue', label = 'Random Prediction Accuracy', linestyle='dashed')
ax.set_xlim(xmin=0, xmax = 1)

# Histogram for Semi-Finalists Prediction Frequency (Cumulative Score)
cm = plt.cm.get_cmap('plasma')
ax2 = plt.subplot(1,2,2)
n, bins, patches = plt.hist(scores2, bins = 5, density = True, edgecolor='black', alpha = 0.9)
bin_centers = 0.5 * (bins[:-1] + bins[1:])
col = bin_centers - min(bin_centers)
col /= max(col)
for c, p in zip(col, patches):
    plt.setp(p, 'facecolor', cm(c))
plt.xlabel("Prediction Accuracy", fontsize = 14)
plt.ylabel("Frequency", fontsize = 14)
plt.title("Semi-Finalists Prediction Frequency: Cumulative Score", fontsize = 13, fontweight = 'bold')
ax2.set_xlim(xmin=0, xmax = 1)
plt.axvline(mean_score2, color = 'black', label = 'Mean Prediction Accuracy', linestyle='dashed')
plt.text(0.6,9.15,'Mean Prediction Accuracy',rotation=0, color = 'black', fontsize = 9.5)
plt.text(0.6,8.75,'Random Prediction Accuracy',rotation=0, color = 'blue', fontsize = 9.5)
plt.axvline(baseline2, color = 'blue', label = 'Random Prediction Accuracy', linestyle='dashed')
ax2.set_xlim(xmin=0, xmax = 1)

plt.subplots_adjust(wspace=0.15)   

plt.show()

# List of annual tournament champions 
tournament_number1team = {}
for year, year_group in tourney_results.groupby('Season'):
    number1team = year_group[(year_group['DayNum'] == 154)]
    tournament_number1team[year] = list(number1team.WTeamID)

# Win Count: top team in the regular season each year
regular_number1team_count = {}
n = 1
for year in year_team_cumulative:
    top_n = dict(sorted(year_team_cumulative[year].items(), key = itemgetter(1), reverse = True)[:n])
    regular_number1team_count[year] = {}
    for team, score in year_team_cumulative[year].items():
        regular_number1team_count[year] = list(top_n.keys()) 

# List of teams and years where win count top team went on to win the championship
reg_tournament_match = {}
for year in regular_number1team_count:
    if tournament_number1team[year] == regular_number1team_count[year]:
        reg_tournament_match[year] = tournament_number1team[year]

# Calculate percentage of years with matching topteam
total_years = len(tournament_number1team)
predict_number1_percent_count = len(reg_tournament_match)/total_years * 100


# Cumulative Score: top team in the regular season each year
regular_number1team_score = {}
n = 1
for year in year_team_cumulative_score:
    top_n_score = dict(sorted(year_team_cumulative_score[year].items(), key = itemgetter(1), reverse = True)[:n])
    regular_number1team_score[year] = {}
    for team, score in year_team_cumulative_score[year].items():
        regular_number1team_score[year] = list(top_n_score.keys()) 

# List of teams and years where cumulative score top team went on to win the championship
reg_tournament_match_score = {}
for year in regular_number1team_score:
    if tournament_number1team[year] == regular_number1team_score[year]:
        reg_tournament_match_score[year] = tournament_number1team[year]

# Calculate percentage of years with matching topteam
predict_number1_percent_score = len(reg_tournament_match_score)/total_years * 100

print("Years where the team with the most number of wins in the regular season became the national champion:")
print(list(reg_tournament_match.keys()))
print()
print("Years where the team with the highest cumulative score over the regular season became the national champion:")
print(list(reg_tournament_match_score.keys()))