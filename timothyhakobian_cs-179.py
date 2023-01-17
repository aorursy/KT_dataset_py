# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pygm179 as gm

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline  
import numpy as np 

import matplotlib.pyplot as plt

import networkx as nx



# Load training data and reduce (subsample) if desired



# Read thru file to get numeric ids for each player 

with open('cs179data/train.csv') as f: lines = f.read().split('\n')



p = 0; playerid = {};

for i in range(len(lines)):

  csv = lines[i].split(',');

  if len(csv) != 10: continue;   # parse error or blank line

  player0,player1 = csv[1],csv[4];

  if player0 not in playerid: playerid[player0]=p; p+=1;

  if player1 not in playerid: playerid[player1]=p; p+=1;



nplayers = len(playerid)

playername = ['']*nplayers

for player in playerid: playername[ playerid[player] ]=player;  # id to name lookup





# Sparsifying parameters (discard some training examples):

pKeep = 1.0   # fraction of edges to consider (immed. throw out 1-p edges)

nEdge = 3     # try to keep nEdge opponents per player (may be more; asymmetric)

nKeep = 5     # keep at most nKeep games per opponent pairs (play each other multiple times)



nplays, nwins = np.zeros( (nplayers,nplayers) ), np.zeros( (nplayers,nplayers) );

for i in range(len(lines)):

  csv = lines[i].split(',');

  if len(csv) != 10: continue;   # parse error or blank line

  a,b = playerid[csv[1]],playerid[csv[4]];

  aw,bw = csv[2]=='[winner]',csv[5]=='[winner]';

  if (np.random.rand() < pKeep):

    if (nplays[a,b] < nKeep) and ( ((nplays[a,:]>0).sum() < nEdge) or ((nplays[:,b]>0).sum() < nEdge) ):

      nplays[a,b] += 1; nplays[b,a]+=1; nwins[a,b] += aw; nwins[b,a] += bw;
games = [

    (0,2, +1),  # P0 played P2 & won

    (0,2, +1),  # played again, same outcome

    (1,2, -1),  # P1 played P2 & lost

    (0,1, -1),  # P0 played P1 and lost

]
nplayers = max( [max(g[0],g[1]) for g in games] )+1

nlevels = 10   # let's say 10 discrete skill levels

scale = .3     # this scales how skill difference translates to win probability



# Make variables for each player; value = skill level

X = [None]*nplayers

for i in range(nplayers):

    X[i] = gm.Var(i, nlevels)   



# Information from each game: what does Pi winning over Pj tell us?

#    Win probability  Pr[win | Xi-Xj]  depends on skill difference of players

Pwin = np.zeros( (nlevels,nlevels) )

for i in range(nlevels):

    for j in range(nlevels):

        diff = i-j                   # find the advantage of Pi over Pj, then 

        Pwin[i,j] = (1./(1+np.exp(-scale*diff)))  # Pwin = logistic of advantage



# before any games, uniform belief over skill levels for each player:

factors = [ gm.Factor([X[i]],1./nlevels) for i in range(nplayers) ]



# Now add the information from each game:

for g in games:

    P1,P2,win = g[0],g[1],g[2]

    if P1>P2: P1,P2,win=P2,P1,-win  # (need to make player IDs sorted...)

    factors.append(gm.Factor([X[P1],X[P2]], Pwin if win>0 else 1-Pwin) )
model = gm.GraphModel(factors)

model.makeMinimal()  # merge any duplicate factors (e.g., repeated games)
if model.nvar < 0:       # for very small models, we can do brute force inference:

    jt = model.joint()

    jt /= jt.sum()       # normalize the distribution and marginalize the table

    bel = [jt.marginal([i]) for i in range(nplayers)] 

else:                    # otherwise we need to use some approximate inference:

    from pyGM.messagepass import LBP, NMF

    lnZ,bel = LBP(model, maxIter=10, verbose=True)   # loopy BP

    #lnZ,bel = NMF(model, maxIter=10, verbose=True)  # Mean field
print("Mean skill estimates: ")

print([ bel[i].table.dot(np.arange(nlevels)) for i in range(nplayers)] )
i,j = 0,1

print("Estimated probability P{} beats P{} next time:".format(i,j))

# Expected value (over skill of P0, P1) of Pr[win | P0-P1]

if i<j:

    print( (bel[i]*bel[j]*gm.Factor([X[i],X[j]],Pwin)).table.sum() )

else:

    print( (bel[i]*bel[j]*gm.Factor([X[i],X[j]],1-Pwin)).table.sum() )

    

# Notes: we should probably use the joint belief over Xi and Xj, but for simplicity

#  with approximate inference we'll just use the estimated singleton marginals