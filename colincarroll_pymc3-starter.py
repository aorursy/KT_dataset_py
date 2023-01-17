import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt

results = pd.read_csv('../input/WRegularSeasonCompactResults.csv')
teams = pd.read_csv('../input/WTeams.csv')
results_2017 = results[results.Season == 2017].reset_index()
teams_2017 = teams[teams.TeamID.isin(results_2017.WTeamID) | teams.TeamID.isin(results_2017.LTeamID)].reset_index(drop=True)
teams_2017['idx'] = teams_2017.index
teams_2017 = teams_2017.set_index('TeamID')
wloc = np.array(teams_2017.loc[results_2017.WTeamID, 'idx']).squeeze()
lloc = np.array(teams_2017.loc[results_2017.LTeamID, 'idx']).squeeze()
with pm.Model() as model:
    team_rating = pm.Normal('rating', mu=0, sd=1, shape=teams_2017.shape[0])
    p = pm.math.sigmoid(team_rating[wloc] - team_rating[lloc])
    # data is organized so the first team always won
    outcome = pm.Bernoulli('outcome_obs', p=p, observed=tt.ones_like(p))
model
with model:
    trace = pm.sample()
pm.traceplot(trace);
top_ten = trace['rating'].mean(axis=0).argsort()[-10:]
teams_2017.set_index('idx').loc[top_ten][-1::-1].reset_index(drop=True)