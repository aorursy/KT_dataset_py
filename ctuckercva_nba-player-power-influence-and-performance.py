import pandas as pd

import statsmodels.api as sm

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.cluster import KMeans

color = sns.color_palette()

from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))

%matplotlib inline
attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv");attendance_valuation_elo_df.head()
salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()

pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()
plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()
br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()


plus_minus_df.rename(columns={"NAME":"PLAYER", "WINS": "WINS_RPM"}, inplace=True)

players = []

for player in plus_minus_df["PLAYER"]:

    plyr, _ = player.split(",")

    players.append(plyr)

plus_minus_df.drop(["PLAYER"], inplace=True, axis=1)

plus_minus_df["PLAYER"] = players

plus_minus_df.head()


nba_players_df = br_stats_df.copy()

nba_players_df.rename(columns={'Player': 'PLAYER','Pos':'POSITION', 'Tm': "TEAM", 'Age': 'AGE', "PS/G": "POINTS"}, inplace=True)

nba_players_df.drop(["G", "GS", "TEAM"], inplace=True, axis=1)

nba_players_df = nba_players_df.merge(plus_minus_df, how="inner", on="PLAYER")

nba_players_df.head()


pie_df_subset = pie_df[["PLAYER", "PIE", "PACE", "W"]].copy()

nba_players_df = nba_players_df.merge(pie_df_subset, how="inner", on="PLAYER")

nba_players_df.head()
salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)

salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)

salary_df.drop(["POSITION","TEAM", "SALARY"], inplace=True, axis=1)

salary_df.head()
diff = list(set(nba_players_df["PLAYER"].values.tolist()) - set(salary_df["PLAYER"].values.tolist()))
len(diff)


nba_players_with_salary_df = nba_players_df.merge(salary_df); 

nba_players_with_salary_df.head()


plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")

corr = nba_players_with_salary_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
sns.lmplot(x="SALARY_MILLIONS", y="WINS_RPM", data=nba_players_with_salary_df)

results = smf.ols('W ~POINTS', data=nba_players_with_salary_df).fit()

print(results.summary())
results = smf.ols('W ~WINS_RPM', data=nba_players_with_salary_df).fit()

print(results.summary())

results = smf.ols('SALARY_MILLIONS ~POINTS', data=nba_players_with_salary_df).fit()

print(results.summary())

results = smf.ols('SALARY_MILLIONS ~WINS_RPM', data=nba_players_with_salary_df).fit()

print(results.summary())

from ggplot import *



p = ggplot(nba_players_with_salary_df,aes(x="POINTS", y="WINS_RPM", color="SALARY_MILLIONS")) + geom_point(size=200)

p + xlab("POINTS/GAME") + ylab("WINS/RPM") + ggtitle("NBA Players 2016-2017:  POINTS/GAME, WINS REAL PLUS MINUS and SALARY")
wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()

wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)

median_wiki_df = wiki_df.groupby("PLAYER").median()



median_wiki_df_small = median_wiki_df[["PAGEVIEWS"]]
median_wiki_df_small = median_wiki_df_small.reset_index()

nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(median_wiki_df_small)

twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()

nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)

nba_players_with_salary_wiki_twitter_df["TEAM"].head()


plt.subplots(figsize=(20,15))

ax = plt.axes()

ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY & TWITTER & WIKIPEDIA)")

corr = nba_players_with_salary_wiki_twitter_df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)
nba_players_with_salary_wiki_twitter_df.head()
def second_largest(numbers):

#returns the second largest number in a vector

    count = 0

    m1 = m2 = float('-inf')

    for x in numbers:

        count += 1

        if x > m2:

            if x >= m1:

                m1, m2 = x, m1            

            else:

                m2 = x

    return m2 if count >= 2 else None
#take the two-team players and assign them to their current team



teams = []

for team in nba_players_with_salary_wiki_twitter_df["TEAM"]:

    try:

        plyr, _ = team.split("""/""")

    except:

        plyr = team

    teams.append(plyr)

nba_players_with_salary_wiki_twitter_df.drop(["TEAM"], inplace=True, axis=1)

nba_players_with_salary_wiki_twitter_df["TEAM"] = teams

nba_players_with_salary_wiki_twitter_df.head()
nba_players_with_salary_wiki_twitter_df[["TEAM", "SALARY_MILLIONS"]].head()
#creates a table with the second highest value for each variable by team

agg_df = nba_players_with_salary_wiki_twitter_df[["TEAM", "SALARY_MILLIONS"]].groupby("TEAM", as_index = False).agg(second_largest)

agg_df.rename(columns = {"SALARY_MILLIONS": "SECOND_HI_SAL"}, inplace= True)

agg_df.head()
max_df = nba_players_with_salary_wiki_twitter_df[["TEAM", "SALARY_MILLIONS"]].groupby("TEAM", as_index = False).agg(max)

max_df.rename(columns = {"SALARY_MILLIONS": "HI_SAL"}, inplace= True)

max_df.head()



top_two_df = max_df.merge(agg_df, how = "inner", on = "TEAM")

top_two_df.head()
p = ggplot(top_two_df,aes(x="HI_SAL", y="SECOND_HI_SAL", label = "TEAM")) + geom_point(size=200)

p + xlab("HIGHEST SALARY") + ylab("2ND HIGHEST SALARY") + ggtitle("NBA Players 2016-2017") + geom_text(hjust = 0.5)

all_player_df = nba_players_with_salary_wiki_twitter_df.merge(agg_df, how="inner", on="TEAM")

all_player_df.rename(columns = {"FG%" : "FGpc",

                                "eFG%": "eFGpc", 

                                "3P":"TRE",

                                "3P%": "TREpc",

                                "3PA":"TREav",

                                "2P%":"DEUCEpc",

                                "2PA":"DEUCEav",

                                "2P":"DEUCE",

                                "FT%" : "FTpc"}, inplace = True)

all_player_df.head()
all_player_df.count()
p = ggplot(all_player_df,aes(x="SECOND_HI_SAL", y="WINS_RPM", color="SALARY_MILLIONS")) + geom_point(size=200)

p + xlab("2ND HIGHEST SALARY") + ylab("WINS/RPM") + ggtitle("NBA Players 2016-2017")
#Structure shamelessly copied from http://planspace.org/20150423-forward_selection_with_statsmodels/

#Modified from an Adjusted R^2 decision criterion to a p-value criterion

#Cannot start column names with numbers, or have % in title



import statsmodels.formula.api as smf



def forward_selected(data, response, power = 0.001):

    """Linear model designed by forward selection.



    Parameters:

    -----------

    data : pandas DataFrame with all possible predictors and response



    response: string, name of response column in data



    Returns:

    --------

    model: an "optimal" fitted statsmodels linear model

           with an intercept

           selected by forward selection

           evaluated by adjusted R-squared

    """

    reiterate = True

    remaining = set(data.columns)

    remaining.remove(response)

    selected = []

    current_score, best_new_score = 0.0, 0.0

    while remaining and reiterate:

        scores_with_candidates = []

        reiterate = False



#finds most significant candidate variable

        

        for candidate in remaining:

            formula = "{} ~ {} + 1".format(response,

                                           ' + '.join(selected + [candidate]))

            score = smf.ols(formula, data).fit().pvalues[candidate]

            scores_with_candidates.append((score, candidate))

        scores_with_candidates.sort(reverse = True)

        best_new_score, best_candidate = scores_with_candidates.pop()

        if best_new_score < power:

            remaining.remove(best_candidate)

            selected.append(best_candidate)

            

#now remove variables with lowered p-value            

            formula = "{} ~ {} + 1".format(response,' + '.join(selected))

            print(formula)

            pvalues = smf.ols(formula, data).fit().pvalues

            print(pvalues)

            dropindices = pvalues[pvalues > power].index

            print(dropindices)

            for dropped in dropindices:

                print(dropped)

                if (dropped != 'Intercept'):

                    selected.remove(dropped)

                    remaining.add(dropped)

            formula = "{} ~ {} + 1".format(response,' + '.join(selected))

            print(formula)



#develop stop condition            



            reiterate = True

    formula = "{} ~ {} + 1".format(response, ' + '.join(selected))

    model = smf.ols(formula, data).fit()

    return model
stepwise = forward_selected (all_player_df.drop(["Rk", "PLAYER", "POSITION", "TEAM"], axis=1), "SALARY_MILLIONS")
print(stepwise.summary())
stepwise = forward_selected (all_player_df.drop(["Rk", "PLAYER", "POSITION", "TEAM", "RPM", "ORPM", "DRPM"], axis=1), "WINS_RPM")
print(stepwise.summary())