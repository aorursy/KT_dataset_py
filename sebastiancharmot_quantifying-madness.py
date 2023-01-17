import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import csv

from scipy.stats import zscore 

import statistics 

import matplotlib.pyplot as plt
loc = ("/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv")

march_loc = ("/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneyCompactResults.csv")

id_name_loc = ("/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeams.csv")



# Keys = team ID, vals = team name

id_to_name = {}



def id_name():

    with open(id_name_loc) as f:

        reader = csv.DictReader(f)

        # maps ID to name

        for row in reader:

            team_id = row["TeamID"]

            id_to_name[team_id] = row["TeamName"]



def massey(year, only_madness):

    id_name()

    teams = set({})

    num_games = 0

    with open(loc) as g:

        reader = csv.DictReader(g)

        for row in reader:

            if int(row['Season']) == year:

                teams.add(id_to_name[row['WTeamID']])

                teams.add(id_to_name[row['LTeamID']])

                num_games += 1

    id_matrix = {}

    matrix_id = 0

    teams = sorted(teams)

    # make teams into an index 

    for team in teams:

        id_matrix[team] = matrix_id

        matrix_id += 1

    # open games file

    # date, team, and score info

    teami = []; scorei = []; teamj = []; scorej = []

    with open(loc) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if int(row['Season']) == year:

                current_i = id_matrix[id_to_name[row['WTeamID']]]

                current_j = id_matrix[id_to_name[row['LTeamID']]]

                s_i = int(row['WScore'])

                s_j = int(row['LScore'])

                # can be used later

                home_away = row['WLoc']

                teami.append(current_i)

                scorei.append(s_i)

                teamj.append(current_j)

                scorej.append(s_j)

    # Massey matrix and point differential vector

    numTeams = max(max(teami),max(teamj)) + 1

    m = np.zeros((numTeams,numTeams))

    p = np.zeros(numTeams)

    # k = 0

    for k in range(len(teami)):

        i = teami[k]

        j = teamj[k]

        # update massey matrix

        m[i,i] += 1; m[j,j] += 1

        m[i,j] += -1; m[j,i] += -1

        # update point differential vector

        p[i] += (scorei[k] - scorej[k])

        p[j] += (scorej[k] - scorei[k])

        # update k

        k += 1

    # solve Massey system

    m[-1,:] = 1

    p[-1] = 0

    r = np.linalg.solve(m,p)



    team_rating = []

    qualified = set({})

    with open(march_loc) as g:

        reader = csv.DictReader(g)

        for row in reader:

            if int(row['Season']) == year:

                qualified.add(id_to_name[row['WTeamID']])

                qualified.add(id_to_name[row['LTeamID']])

    for i in range(len(r)):

        if only_madness:

            if teams[i] in qualified:

                team_rating.append([teams[i],r[i]])

        else:

            team_rating.append([teams[i],r[i]])

    team_rating = sorted(team_rating, key = lambda x: x[1], reverse= True)



    return team_rating



massey(2018, True)
loc = ("/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MRegularSeasonCompactResults.csv")

march_loc = ("/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MNCAATourneyCompactResults.csv")

id_name_loc = ("/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1/MTeams.csv")



# Keys = team ID, vals = team name

id_to_name = {}



name_to_id = {}



def id_name():

    with open(id_name_loc) as f:

        reader = csv.DictReader(f)

        # maps ID to name

        for row in reader:

            team_id = row["TeamID"]

            id_to_name[team_id] = row["TeamName"]

            name_to_id[row["TeamName"]] = int(row["TeamID"])

            

def massey(year, only_madness):

    teams = set({})

    num_games = 0

    with open(loc) as g:

        reader = csv.DictReader(g)

        for row in reader:

            if int(row['Season']) == year:

                teams.add(id_to_name[row['WTeamID']])

                teams.add(id_to_name[row['LTeamID']])

                num_games += 1

            if int(row['Season']) > year:

                break

    id_matrix = {}

    matrix_id = 0

    teams = sorted(teams)

    # make teams into an index 

    for team in teams:

        id_matrix[team] = matrix_id

        matrix_id += 1

    # open games file

    # date, team, and score info

    teami = []; scorei = []; teamj = []; scorej = []

    with open(loc) as f:

        reader = csv.DictReader(f)

        for row in reader:

            if int(row['Season']) == year:

                current_i = id_matrix[id_to_name[row['WTeamID']]]

                current_j = id_matrix[id_to_name[row['LTeamID']]]

                s_i = int(row['WScore'])

                s_j = int(row['LScore'])

                # can be used later

                home_away = row['WLoc']

                teami.append(current_i)

                scorei.append(s_i)

                teamj.append(current_j)

                scorej.append(s_j)

    # Massey matrix and point differential vector

    numTeams = max(max(teami),max(teamj)) + 1

    m = np.zeros((numTeams,numTeams))

    p = np.zeros(numTeams)

    # k = 0

    for k in range(len(teami)):

        i = teami[k]

        j = teamj[k]

        # update massey matrix

        m[i,i] += 1; m[j,j] += 1

        m[i,j] += -1; m[j,i] += -1

        # update point differential vector

        p[i] += (scorei[k] - scorej[k])

        p[j] += (scorej[k] - scorei[k])

        # update k

        k += 1

    # solve Massey system

    m[-1,:] = 1

    p[-1] = 0

    r = np.linalg.solve(m,p)



    team_ratings = []

    team_ids = []

    qualified = set({})

    with open(march_loc) as g:

        reader = csv.DictReader(g)

        for row in reader:

            if int(row['Season']) == year:

                qualified.add(id_to_name[row['WTeamID']])

                qualified.add(id_to_name[row['LTeamID']])

    for i in range(len(r)):

        if only_madness:

            if teams[i] in qualified:

                team_ratings.append(r[i])

                team_ids.append(name_to_id[teams[i]])



    team_zscores = zscore(team_ratings)



    """ Cinderella Analysis """



    # maps team ids to their massey ratings zscores

    id_zscore = {}



    for i in range(len(team_ids)):

        id_zscore[team_ids[i]] = team_zscores[i]



    # 32 teams 

    day_138_139 = []

    weight_32 = 1/31    

    # 16 teams 

    day_143_144 = []

    weight_16 = 2/31

    # 8 teams 

    day_145_146 = []

    lowest_145_146 = 100

    weight_8 = 4/31

    # 4 teams 

    day_152 = []

    lowest_152 = 100

    weight_4 = 8/31

    # 2 teams final 

    day_154 = []

    lowest_154 = 100

    weight_2 = 16/31



    with open(march_loc) as g:

        reader = csv.DictReader(g)

        for row in reader:

            if int(row['Season']) == year:

                current_day = row['DayNum']

                team_1 = int(row['WTeamID'])

                team_2 = int(row['LTeamID'])

                if current_day == '138' or current_day == '139':

                    day_138_139.append(id_zscore[team_1])

                    day_138_139.append(id_zscore[team_2])

                elif current_day == '143' or current_day == '144':

                    day_143_144.append(id_zscore[team_1])

                    day_143_144.append(id_zscore[team_2])

                elif current_day == '145' or current_day == '146':

                    day_145_146.append(id_zscore[team_1])

                    day_145_146.append(id_zscore[team_2])

                    lowest = min(lowest_145_146,id_zscore[team_1],id_zscore[team_2])

                    lowest_145_146 = lowest

                elif current_day == '152':

                    day_152.append(id_zscore[team_1])

                    day_152.append(id_zscore[team_2])

                    lowest = min(lowest_152,id_zscore[team_1],id_zscore[team_2])

                    lowest_152 = lowest

                elif current_day == '154':

                    day_154.append(id_zscore[team_1])

                    day_154.append(id_zscore[team_2])

                    lowest = min(lowest_154,id_zscore[team_1],id_zscore[team_2])

                    lowest_154 = lowest



    weighted_zscore_average = 0

    weighted_zscore_average += weight_32 * statistics.mean(day_138_139)

    weighted_zscore_average += weight_16 * statistics.mean(day_143_144)

    weighted_zscore_average += weight_8 * statistics.mean(day_145_146)

    weighted_zscore_average += weight_4 * statistics.mean(day_152)

    weighted_zscore_average += weight_2 * statistics.mean(day_154)



    return [weighted_zscore_average, lowest_154]



def plot_weighted_average():

    id_name()

    n_groups = 0

    weighted_zscore = []

    Lowest_Zscore_Final  = []

    x_labels = []



    for year in range(1985,2020):

        r = massey(year,True)

        n_groups += 1

        weighted_zscore.append(r[0])

        Lowest_Zscore_Final.append(r[1])

        x_labels.append("'" + str(year)[-2:])



    # create plot

    fig, ax = plt.subplots()

    index = np.arange(n_groups)

    bar_width = 0.35

    opacity = 0.8



    rects1 = plt.bar(index, weighted_zscore, bar_width,

    alpha=opacity,

    color='b',

    label='Weighted Z-Score')



    rects2 = plt.bar(index + bar_width, Lowest_Zscore_Final , bar_width,

    alpha=opacity,

    color='g',

    label='Finals Lowest Zscore')



    plt.xlabel('Year')

    plt.ylabel('Z-Score')

    plt.title('Z-Scores vs. Year (Men\'s)')

    plt.xticks(index + bar_width, x_labels)

    plt.legend()



    plt.tight_layout()

    plt.show()



# plot_weighted_average()