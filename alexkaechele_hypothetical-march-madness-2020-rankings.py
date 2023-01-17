# Load needed packages

import numpy as np

import pandas as pd

import pystan

import matplotlib.pyplot as plt

import random



# Import data

dat = pd.read_csv("/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MRegularSeasonCompactResults.csv") 

team_key = pd.read_csv("/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/MTeams.csv")[["TeamID", "TeamName"]]



# Filter to 2019

dat = dat[dat.Season == 2020].reset_index(drop = True)



# Make home

dat['homei'] = np.where(dat.WLoc == "H", 1, 0)

dat['homej'] = np.where(dat.WLoc == "A", 1, 0)



# Create margin

dat['margin'] = dat.WScore - dat.LScore



# Filter to needed columns and rename

dat = dat[["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "margin", "homei", "homej"]]

dat = dat.rename(columns={'WTeamID' : 'teami',

                          'WScore'  : 'scorei', 

                          'LTeamID' : 'teamj', 

                          'LScore'  : 'scorej',

                          'DayNum'  : 'daynum',

                          'Season'  : 'season'})



# Create a game id

dat["gameid"] = np.where(dat['teami'] < dat['teamj'], 

                         dat['teami'].astype(str) + "_" + dat['teamj'].astype(str), 

                         dat['teamj'].astype(str) + "_" + dat['teami'].astype(str))



# Set up team id mapping

team_key["id"] = range(1, len(team_key.index) + 1)



# Recoding ids to be between 1 and 366

dat = dat.merge(team_key, left_on="teami" , right_on="TeamID")

dat = dat.drop(columns=["TeamName", "teami", "TeamID"])

dat = dat.rename(index = str, columns = {"id" : "teami"})

dat = dat.merge(team_key, left_on="teamj" , right_on="TeamID")

dat = dat.drop(columns=["TeamName", "teamj", "TeamID"])

dat = dat.rename(index = str, columns = {"id" : "teamj"})



# Final dataset for modeling

names = ["N", "y", "h_i", "h_j", "team_i", "team_j", "N_g"]

values = [len(dat.index), dat.margin, dat.homei, dat.homej, dat.teami, dat.teamj, 367]



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

    eta ~ normal(4,1);

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

theta = pd.DataFrame(fit.extract()["theta"])

alpha = pd.DataFrame(fit.extract()["alpha"])

sigma = fit.extract()["sigma"]

alpha.columns = team_key.TeamName

theta.columns = team_key.TeamName



# Filtering to top 25 teams

theta25 = theta[theta.median().nlargest(25).index]

theta25 = theta25[theta25.columns[::-1]]



# Creating the plot

theta25.boxplot(grid = False, vert = False, showfliers = False, figsize=(12, 8))

plt.title('Team Power Rankings')

plt.xlabel('Skill Level')

plt.ylabel('Teams')