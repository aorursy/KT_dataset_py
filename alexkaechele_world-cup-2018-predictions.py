# Importing libraries

import numpy as np

import pandas as pd

import pystan

import matplotlib.pyplot as plt

import random

import warnings

warnings.filterwarnings("ignore")



# Loading data

dat = pd.read_csv("../input/results.csv")



# Splitting train and test data

train = dat[(pd.to_datetime(dat.date) < pd.to_datetime("14-Jun-2018")) & (pd.to_datetime(dat.date) > pd.to_datetime("2010"))] # Need to think about

test = dat[(dat.tournament == "FIFA World Cup") & (dat.country == "Russia")]



# Create home and away

train["homei"] = 1 - train["neutral"]

train["homej"] = 0



# Create margin

train["margin"] = train["home_score"] - train["away_score"]



# Filter to needed columns and rename

train = train[["date", "home_team", "home_score", "away_team", "away_score", "margin", "homei", "homej"]]

train = train.rename(columns={'home_team'  : 'teami',

                          'home_score' : 'scorei', 

                          'away_team'  : 'teamj', 

                          'away_score' : 'scorej'})



# Create a game id

train["gameid"] = train['teami'].astype(str) + "_" + train['teamj']



# Set up team id mapping

team_key = pd.DataFrame(np.array([train.teami.append(train.teamj).unique()]).transpose(),

                        columns = ['teamname']).reset_index()

team_key["index"] = team_key["index"] + 1



# Recoding ids to be between 1 and 276

train = train.merge(team_key, left_on = "teami" , right_on = "teamname")

train = train.drop(columns = ["teami", "teamname"])

train = train.rename(index = str, columns = {"index" : "teami"})

train = train.merge(team_key, left_on = "teamj" , right_on = "teamname")

train = train.drop(columns = ["teamj", "teamname"])

train = train.rename(index = str, columns = {"index" : "teamj"})



# Final dataset for modeling

names = ["N", "y", "h_i", "h_j", "team_i", "team_j", "N_g"]

values = [len(train.index), train.margin, train.homei, train.homej, train.teami, train.teamj, 276]



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

    eta ~ normal(.33,1);

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

alpha.columns = team_key.teamname

theta.columns = team_key.teamname



# Filtering to top 25 teams

theta25 = theta[theta.median().nlargest(25).index]

theta25 = theta25[theta25.columns[::-1]]



# Creating the plot

theta25.boxplot(grid = False, vert = False, showfliers = False, figsize=(12, 8))

plt.title('Team Power Rankings (2010 to Tournament)')

plt.xlabel('Skill Level')

plt.ylabel('Teams')
# Setting seed

random.seed(1865)



# Defining compare function

def compare(i, j, th= theta, a = alpha, sig = sigma, homei = 0, homej = 0, reps = 1000):

    win_prob = []

    

    # Simulating specified number of games

    for r in range(1, reps):

        win_prob.append(

            np.mean(

                

                # Ability difference

                th[i] - th[j] +

                

                # Adjusting for home court

                a[i]*homei - a[j]*homej +

                

                # Team performance variance

                np.random.normal(0, 

                                 sig[random.randrange(len(sig))], 

                                 len(th.index)

                ) > 0

            )

        )

    

    # Averaging game results

    win_prob = np.mean(win_prob)

    

    # Displaying results

    print(i + " has a " + str(round(win_prob*100, 2)) + "% chance of beating " + j)

    

# Looking at final game

compare("France", "Croatia")