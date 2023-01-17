# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import networkx as nx

import itertools

import matplotlib.pyplot as plt # nx won't draw without this

%matplotlib notebook 
dataPath = '../input/' # set the data path
# read songs into pandas data frame

songs = pd.read_csv(dataPath + 'songs.csv')

songs = songs[['song_id','genre_ids']] # we don't need the rest
# convert genre ids to list

songs['genre_ids'] = songs['genre_ids'].map(lambda x: [int(y) for y in str(x).split('|')] if not pd.isnull(x) else [])
# get unique list of genre ids

genres = songs['genre_ids'].values.tolist()

genres = [j for i in genres for j in i]

genres = list(set(genres))
# number of unique genres

numGenres = len(genres)

print("There are %s unique genres." %numGenres)
# init genre similarity matrix S and genre score list

S = np.zeros((numGenres,numGenres))

scores = np.zeros((numGenres,))
# read user listening data

listen = pd.read_csv(dataPath + 'train.csv')
# we only consider songs with target 1

listen = listen[listen['target'] == 1]

listen = listen[['msno','song_id']] # we don't need the rest
# join the two datasets

songs.set_index('song_id', inplace=True)

df = listen.join(songs, how="left", on="song_id")

df.dropna(axis=0,inplace=True) # drop anything with missing data
# group by user and process groups

for user, frame in df.groupby('msno'):

    userGenres = frame['genre_ids'].values.tolist() # get all the genres liked by this user

    userGenres = [j for i in userGenres for j in i] # convert to a single list

    userGenres = set(list(userGenres)) # take only unique values



    for aGenre in userGenres: # increase genre score

        m = genres.index(aGenre) 

        scores[m] += 1



    combs = itertools.combinations(userGenres, 2) # increase co-occurrence scores in matrix S

    for comb in combs:

        S[genres.index(comb[0]),genres.index(comb[1])] += 1

        S[genres.index(comb[1]),genres.index(comb[0])] += 1
G = nx.Graph()
for g in genres: # add nodes

    G.add_node(g)
for i,gI in enumerate(genres): # add edges

    for j,gJ in enumerate(genres):

        if gJ >= gI:

            continue

        if S[i][j] > 0:

            G.add_edge(gI,gJ,weight=S[i][j])
# filter out nodes with score < 1000

nodeList = [x for i,x in enumerate(genres) if scores[i] > 1000]

G2 = G.subgraph(nodeList)

nodeSizes = [0.1 * x for x in scores if x > 1000]
nx.draw_spectral(G2, with_labels=True, node_size=nodeSizes, alpha=0.2, width=0.1, random_state=1985) # draw the new graph

plt.show() # there are some warnings that seem to come from nx interacting with matplotlib