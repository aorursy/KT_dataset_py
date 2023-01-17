#!pip install arviz

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error

import seaborn as sbn

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
import theano



"""

These are the functions that we will require if we want to compute the predictions

and Elo ratings for teams by hand (traditional method). You can find these equations

and their explanation here: http://dataskeptic.com/blog/methods/2017/calculating-an-elo-rating

"""



def Logistic_Rating(rating):

    return 10.0**(rating/400)



def Expected(rating1, rating2):

    """

    These should be Logistic Ratings

    """

    log_rating1 = Logistic_Rating(rating1)

    log_rating2 = Logistic_Rating(rating2)

    return log_rating1/(log_rating1+log_rating2)



def Update_Rating(rating1, K, S, E):

    """

    rating1: rating should be the r

    K : Scaling factor between 1,99

    S : Outcome of game [0,1]

    E : Expected Outcome of the game

    """

    return rating1 + K*(S-E)
"""

Load in the data set of teams and their overall win/loss fraction for a match.

Each match consists of a number of games, and the overall fraction represents the

win rate of the first team. If a match between Team A and Team B resulted in

1 win for Team A and 3 wins for Team B, the data would look like this:

====================

Team A, 0.25, Team B

"""



regular_season = pd.read_csv('../input/regular_season_frac_score.csv')

regular_season.head()                             
"""

For each team, let us make an index for them.

"""



teams = regular_season.Team1.unique() 

n_teams = len(teams)

teams = pd.DataFrame({"index":range(len(teams)), "team": teams})

teams.head(20)
"""

Join the team index with the record values.

"""



regular_season = pd.merge(regular_season,\

                    teams.rename(index=str,\

                    columns={"team": "Team1", "index": "team1_index"}),

                    on = ['Team1'] )



regular_season = pd.merge(regular_season,\

                    teams.rename(index=str,\

                    columns={"team": "Team2", "index": "team2_index"}),

                    on = ['Team2'] )



regular_season = regular_season.drop(['Team1','Team2'], axis=1)



regular_season.head()
import theano.tensor as T



"""

In PyMC3, we can enforce the model data to be in this 

shared tensor format Theano provides. That way we can do

train and evaluation splits effectively.

"""



winner_index = theano.shared(np.array(regular_season['team1_index']))

loser_index = theano.shared(np.array(regular_season['team2_index']))

model_output = theano.shared(np.array(regular_season['Record']))
import pymc3 as pm



print('Running on PyMC3 v{}'.format(pm.__version__))



elo = pm.Model()



with elo:

    elo_team = pm.Normal("elo_team", mu = 2000.0, sd=1000.0, shape=n_teams)

    

    """

    We can convert the Elo ratings to a logistic rating. This enables us to

    make predictions on individual matchup outcomes.

    """

    log_rating2 = 10.0**(elo_team[loser_index]/400.0)

    log_rating1 = 10.0**(elo_team[winner_index]/400.0)

    E = log_rating1/(log_rating1+log_rating2) # Expected value

    error =  pm.HalfCauchy('error', beta=1.0)

    

    """

    The Expectation value is a value between 0 and 1, with

    0 predicting that Team 1 has a 0% chance of winning the matchup.

    """    

    out = pm.Normal('out', mu=E, sd = error, observed=model_output)
"""

You can choose to use ADVI or NUTS sampler. As this 

is such a small dataset, the NUTS works fine.

"""



use_advi = False

with elo:

    if use_advi:

        inference = pm.ADVI()

        advi_approx = pm.fit(n=100000, method=inference)

    else:

        trace = pm.sample(8000)
#Now we sample from our approximation in order to get a similar trace

if use_advi:

    trace = advi_approx.sample(10000)
"""

What do our ratings look like? How are they distributed?



"""



pm.traceplot(trace);
pm.summary(trace)
"""

Two helper functions to either make prediction probabilities for some inputs,

or to score the model on a set of target values.



You can think of these utilities as sampling from the _distribution_ of 

possible Elo ratings for each team, instead of a single value. If our model 

is very uncertain of a rating for a given team, their prediction value will be

reflective of that.

"""



def make_preds(trace,model_name):

    ppc = pm.sample_posterior_predictive(trace, model=model_name, samples=500)

    return ppc['out'].mean(axis=0)



def scoreModel(trace,y,model_name):

    ppc = pm.sample_posterior_predictive(trace, model=model_name, samples=2000)

    pred = ppc['out'].mean(axis=0)

    print ("RMSE: %0.3f" %(  np.sqrt(mean_squared_error([round(rec) for rec in y],[round(x) for x in pred]))))

    print ("ROC AUC: %0.3f" %(roc_auc_score([round(rec) for rec in y], pred)))

    print ("Accuracy: %0.3f" %(accuracy_score([round(rec) for rec in y],[round(x) for x in pred]))) 



scoreModel(trace,regular_season['Record'],elo)
"""

Unfortunately there are very few playoff matches,

so our model will have to be very good on a small 

number of samples.

"""



playoffs = pd.read_csv('../input/playoffs_frac_score.csv')

print (playoffs.count()[0])

playoffs.tail()
playoffs = pd.merge(playoffs,\

                    teams.rename(index=str,\

                    columns={"team": "Team1", "index": "team1_index"}),

                    on = ['Team1'] )



playoffs = pd.merge(playoffs,\

                    teams.rename(index=str,\

                    columns={"team": "Team2", "index": "team2_index"}),

                    on = ['Team2'] )



playoffs = playoffs.drop(['Team1','Team2'], axis=1)

playoffs.head()
"""

Initialize our evaluation data set in the shared tensor.

"""



winner_index.set_value(np.array(playoffs['team1_index']))

loser_index.set_value(np.array(playoffs['team2_index']))

model_output.set_value(np.array(playoffs['Record']))
"""

Our model does not do that well. Our RMSE of our 

Expected value is pretty poor, and our ROC is just above 

random guessing. 

"""



scoreModel(trace,playoffs['Record'],elo)
"""

If we examine our predictions in more detail, we can see that 

our model is not confident at all. Most matches end fairly 

decisively, but our model generally only gives 50% odds

to any given outcome.

"""



playoffs['predictions'] = make_preds(trace,elo)

playoffs
"""

Using the mean value of each Elo rating that our model gives us,

we can make raw predictions as well.

"""



playoffs = pd.merge(playoffs, wins[['index','Elo']].rename(index=str,columns={'index':"team1_index"}), on=['team1_index'],how='inner')

playoffs = playoffs.rename(index=str,columns={'Elo': 'team1_elo'})

playoffs = pd.merge(playoffs, wins[['index','Elo']].rename(index=str,columns={'index':"team2_index"}), on=['team2_index'],how='inner')

playoffs = playoffs.rename(index=str,columns={'Elo': 'team2_elo'})

playoffs.head()
"""

As expected, this does not affect our metrics to any significance.

"""



playoffs['raw_expected'] = Expected(playoffs['team1_elo'],playoffs['team2_elo'])

playoffs.tail()
print ("ROC AUC: %0.3f" %(roc_auc_score([round(rec) for rec in playoffs['Record']], playoffs['raw_expected'])))

print ("Accuracy: %0.3f" %(accuracy_score([round(rec) for rec in playoffs['Record']],[round(x) for x in playoffs['raw_expected']]))) 