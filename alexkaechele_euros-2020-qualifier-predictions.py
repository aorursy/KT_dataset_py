# Load packages

import pandas as pd

import numpy as np

import pystan

import matplotlib.pyplot as plt

import random



# Import data

dat = pd.read_csv("../input/results.csv")

euros = dat.loc[(dat.tournament =="UEFA Euro qualification") & (pd.DatetimeIndex(dat['date']).year == 2019)]



# Take last 10 years of data

dat = dat[ pd.DatetimeIndex(dat['date']).year >= 2010]



# Create home and away

dat["homei"] = 1-dat.neutral

dat["homej"] = 0



# Create margin

dat['margin'] = dat.home_score - dat.away_score





# Set up team id mapping

team_key = pd.DataFrame({"teamname" : dat.home_team.append(dat.away_team).unique(),

                         "teamid"   : range(1, 1+len(dat.home_team.append(dat.away_team).unique()))})



# Recoding ids in qualifying data

dat = dat.merge(team_key, left_on="home_team" , right_on="teamname")

dat = dat.drop(columns=["teamname", "home_team"])

dat = dat.rename(index = str, columns = {"teamid" : "home_team"})



dat = dat.merge(team_key, left_on="away_team" , right_on="teamname")

dat = dat.drop(columns=["teamname", "away_team"])

dat = dat.rename(index = str, columns = {"teamid" : "away_team"})





# Final dataset for modeling

names = ["N", "y", "h_i", "h_j", "team_i", "team_j", "N_g"]

values = [len(dat.index), dat.margin, dat.homei, dat.homej, dat.home_team, dat.away_team, 278]



train = dict(zip(names, values))
model = """

data {

    int N;

    vector[N] y;

    int team_i[N];

    int team_j[N];

    int h_i[N];

    int h_j[N];

    int N_g;

}

parameters {

    vector[N_g] alpha_raw;

    vector[N_g] theta_raw;

    real eta;

    real<lower=0> tau_theta;

    real<lower=0> tau_alpha;

    real<lower=0> sigma;

}

transformed parameters {

    vector[N_g] alpha;

    vector[N_g] theta;

    alpha = eta + alpha_raw*tau_alpha;

    theta = theta_raw*tau_theta;

}

model {

    // vector for conditional mean storage

    vector[N] mu;



    // priors

    tau_theta ~ cauchy(0,1)T[0,];

    tau_alpha ~ cauchy(0,.25)T[0,];

    sigma ~ cauchy(0,1)T[0,];

    eta ~ normal(.5,.25);

    theta_raw ~ normal(0,1);

    alpha_raw ~ normal(0,1);



    // define mu for the Gaussian

    for( t in 1:N ) {

    mu[t] = (theta[team_i[t]] + alpha[team_i[t]]*h_i[t]) - 

    (theta[team_j[t]] + alpha[team_j[t]]*h_j[t]);

}



    // the likelihood

    y ~ normal(mu,sigma);

}

"""



sm = pystan.StanModel(model_code = model)

fit = sm.sampling(data = train, 

                  iter = 1500, 

                  warmup = 750,

                  refresh = 100,

                  control = dict(adapt_delta = 0.9))
# Extracting team skill levels

th = pd.DataFrame(fit.extract()["theta"])

a = pd.DataFrame(fit.extract()["alpha"])

sig = fit.extract()["sigma"]

a.columns = team_key.teamname

th.columns = team_key.teamname



# Filtering to top 25 teams

theta25 = th[th.median().nlargest(25).index]

theta25 = theta25[theta25.columns[::-1]]



# Creating the plot

theta25.boxplot(grid = False, vert = False, showfliers = False, figsize=(12, 8))

plt.title('Team Power Rankings')

plt.xlabel('Skill Level')

plt.ylabel('Teams')
# Groups

GroupA = ["England", "Bulgaria", "Kosovo", "Montenegro", "Czech Republic"]

GroupB = ["Ukraine", "Luxembourg", "Portugal", "Serbia", "Lithuania"]

GroupC = ["Northern Ireland", "Germany", "Netherlands", "Estonia", "Belarus"]

GroupD = ["Ireland", "Switzerland", "Denmark", "Gibraltar", "Georgia"]

GroupE = ["Slovakia", "Wales", "Hungary", "Croatia", "Azerbaijan"]

GroupF = ["Spain", "Sweden", "Romania", "Malta", "Norway", "Faroe Islands"]

GroupG = ["Poland", "Israel", "Macedonia", "Slovenia", "Austria", "Latvia"]

GroupH = ["France", "Turkey", "Albania", "Iceland", "Andorra", "Moldova"]

GroupI = ["Belgium", "Russia", "Kazakhstan", "Cyprus", "Scotland", "San Marino"]

GroupJ = ["Italy", "Greece", "Bosnia-Herzegovina", "Finland", "Armenia", "Liechtenstein"]



def matchups(group):

    df = pd.DataFrame({'home_team': [], 'away_team': []})

    for i in group:

        for j in [x for x in group if x != i]:

            df = df.append({'home_team': i, 'away_team': j}, ignore_index=True)

    return df

        



def get_current_table(group, data):

    group = matchups(group)

    

    euro_results = group.merge(data, on = ["home_team", "away_team"], how = "left")

    return euro_results[["home_team", "away_team", "home_score", "away_score"]]







def get_standings(group, data):

    

    # Getting results

    current_table = get_current_table(group, data)

    current_table["GD"] = current_table.home_score - current_table.away_score

        

    # GF, GA, and record

    standings = pd.DataFrame(current_table.groupby(["home_team"])["home_score"].sum() + current_table.groupby(["away_team"])["away_score"].sum())#.reset_index()

    standings["goals_against"] = current_table.groupby(["home_team"])["away_score"].sum() + current_table.groupby(["away_team"])["home_score"].sum()

    standings[["win", "draw", "loss"]] = current_table.groupby('home_team').GD.apply(lambda x: pd.Series([(x > 0).sum(), (x == 0).sum(), (x < 0).sum()])).unstack() + current_table.groupby('away_team').GD.apply(lambda x: pd.Series([(x < 0).sum(), (x == 0).sum(), (x > 0).sum()])).unstack()

    

    # Formatting

    standings = standings.reset_index()

    standings.columns = ["team", "goals_for", "goals_against", "win", "draw", "loss"]

    

    # GD, points

    standings["goal_diff"] = standings.goals_for - standings.goals_against

    standings["points"] = standings.win*3 + standings.draw

    

    # Ordering standings

    standings = standings.sort_values(by = ["points", "goal_diff", "goals_for"], ascending =False)

    standings.index = range(1,len(standings.index)+1)

    

    return standings
def short_compare(i, j, homei =1, homej=0, th= th, a = a, sig = sig, allowdraw = True):

    

    home = int(th[i].sample(1).values + a[i].sample(1).values*homei)

    away = int(th[j].sample(1).values + a[j].sample(1).values*homej)

       

    return home, away

    

short_compare("Bosnia-Herzegovina", "Germany")



def short_sim_season(group, data):

    current_table = get_current_table(group, data)

    

    current_table_1 = current_table[current_table.home_score.isnull()].copy()

    

    for i in current_table_1.index:

        current_table["home_score"][i], current_table["away_score"][i] = short_compare(current_table.home_team[i], current_table.away_team[i])

    

    current_table["GD"] = current_table.home_score - current_table.away_score

        

    # GF, GA, and record

    standings = pd.DataFrame(current_table.groupby(["home_team"])["home_score"].sum() + current_table.groupby(["away_team"])["away_score"].sum())#.reset_index()

    standings["goals_against"] = current_table.groupby(["home_team"])["away_score"].sum() + current_table.groupby(["away_team"])["home_score"].sum()

    standings[["win", "draw", "loss"]] = current_table.groupby('home_team').GD.apply(lambda x: pd.Series([(x > 0).sum(), (x == 0).sum(), (x < 0).sum()])).unstack() + current_table.groupby('away_team').GD.apply(lambda x: pd.Series([(x < 0).sum(), (x == 0).sum(), (x > 0).sum()])).unstack()

    

    # Formatting

    standings = standings.reset_index()

    standings.columns = ["team", "goals_for", "goals_against", "win", "draw", "loss"]

    

    # GD, points

    standings["goal_diff"] = standings.goals_for - standings.goals_against

    standings["points"] = standings.win*3 + standings.draw

    

    # Ordering standings

    standings = standings.sort_values(by = ["points", "goal_diff", "goals_for"], ascending =False)

    standings.index = range(1,len(standings.index)+1)

    

    return standings





def long_sim_season(group, data, reps = 1000):

    pd.options.mode.chained_assignment = None  # default='warn'

    

    qual = pd.Series()

    for i in range(0, reps):

        season = short_sim_season(group, data)

        qual = qual.append(season["team"].head(2))

    return qual.value_counts()/reps*100
# Current Standings of Group

get_standings(GroupA, euros)
# Odds of each team qualifying under simulation

long_sim_season(GroupA, euros)
# Current Standings of Group

get_standings(GroupB, euros)
# Odds of each team qualifying under simulation

long_sim_season(GroupB, euros)
# Current Standings of Group

get_standings(GroupC, euros)
# Odds of each team qualifying under simulation

long_sim_season(GroupC, euros)
# Current Standings of Group

get_standings(GroupD, euros)
# Odds of each team qualifying under simulation

long_sim_season(GroupD, euros)
# Current Standings of Group

get_standings(GroupE, euros)
# Odds of each team qualifying under simulation

long_sim_season(GroupE, euros)
# Current Standings of Group

get_standings(GroupF, euros)
# Odds of each team qualifying under simulation

long_sim_season(GroupF, euros)
# Current Standings of Group

get_standings(GroupG, euros)
# Odds of each team qualifying under simulation

long_sim_season(GroupG, euros)
# Current Standings of Group

get_standings(GroupH, euros)
# Odds of each team qualifying under simulation

long_sim_season(GroupH, euros)
# Current Standings of Group

get_standings(GroupI, euros)
# Odds of each team qualifying under simulation

long_sim_season(GroupI, euros)
# Current Standings of Group

get_standings(GroupJ, euros)
# Odds of each team qualifying under simulation

long_sim_season(GroupJ, euros)