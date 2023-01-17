import numpy as np
import pandas as pd
from pandas import Series, DataFrame

import scipy.stats as st
# Parameters:

num_tourneys=4  # number of tournaments
num_games=num_tourneys*67 # total number of games in dataset
mu, sigma = 0, 1 # mean and standard deviation for prediction
num_rows=10000 # length of results table
normal_dist = True # normally distributed predictions if True, uniform predictions if False

# Create the outcome data frame.
df=pd.DataFrame(np.zeros((num_games,4)),columns=['Pred','Random','Outcome','Score'])

# Pred = "perfect" predicted probability, normally distributed
# Random = uniformly distributed random number
# Outcome = 1 if Pred>Random, else 0
# Score = the Log Loss calculation

#create the results data frame to record LogLoss for each iteration.
df_results=pd.DataFrame(np.zeros(num_rows),columns=['LogLoss'])
# generate num_rows sets of random numbers, determine outcomes, calculate LogLoss

for row in range(num_rows):
    
    if normal_dist:
        df.Pred=st.norm.cdf(np.random.randn(num_games))
    else:
        df.Pred=np.random.rand(num_games)
    
    df.Random=np.random.rand(num_games)
    df.Outcome=np.where(df.Pred>df.Random,1,0)
    df.Score=df.Outcome*np.log(df.Pred)+(1-df.Outcome)*np.log(1-df.Pred)
    df_results.iloc[row,0]=-df.Score.mean()
    
print(df_results.describe())


