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
import numpy as np

import pandas as pd



data = pd.read_csv("/kaggle/input/qb-ratings/2.csv")

data_filter = data[["Player", "Cmp%", "Yds", "TD", "Int"]]

data_filter
mean_col = data_filter.agg({'Cmp%': [np.mean], 'Yds': [np.mean], 'TD': [np.mean], 'Int': [np.mean]})



mean_col
data_filter.Yds = data_filter.Yds.astype(float)

data_filter.TD = data_filter.TD.astype(float)

data_filter.Int = data_filter.Int.astype(float)

data_filter
for i in data_filter.index:

    data_filter["Cmp%"][i] = abs(data_filter["Cmp%"][i]-mean_col["Cmp%"])/mean_col["Cmp%"]

    data_filter["Yds"][i] = (data_filter["Yds"][i]-mean_col["Yds"])/(mean_col["Yds"]+data_filter["Yds"][i])

    data_filter["TD"][i] = (data_filter["TD"][i]-mean_col["TD"])/(data_filter["TD"][i]+mean_col["TD"])

    data_filter["Int"][i] = (data_filter["Int"][i]-mean_col["Int"])/(mean_col["Int"]+data_filter["Int"][i])



data_filter    
player_dict = data_filter.T.to_dict('list')

player_dict[3][0]
from scipy import spatial



def ComputeDistance(a, b):

    playerA = np.sqrt(a[1]**2 + a[2]**2 + a[3]**2)

    playerB = np.sqrt(b[1]**2 + b[2]**2 + b[3]**2)

    dist = ((playerA - playerB)**2)/abs(playerA - playerB)

    return dist

    

ComputeDistance(player_dict[1], player_dict[6])
import operator

playerID = data_filter["Player"]



def getNeighbors(playerID, K):

    distances = []

    for player in player_dict:

        if (player != playerID):

            distance = ComputeDistance(player_dict[playerID], player_dict[player])

            distances.append((player, distance))

    distances.sort(key=operator.itemgetter(1))

    neighbors = []

    for x in range(K):

        neighbors.append(distances[x][0])

    return neighbors



K = 10

measured_dist = 0

neighbors = getNeighbors(1, K)

for neighbor in neighbors:

    measured_dist += player_dict[neighbor][3]

    print (player_dict[neighbor][0])

    print(str(player_dict[neighbor][1]) + " " + str(player_dict[neighbor][2]) + " " + str(player_dict[neighbor][3]) + " " + str(player_dict[neighbor][4]) )

    

#measured_dist /= K
display(ComputeDistance(player_dict[1], player_dict[38]))  ###Tannehill

ComputeDistance(player_dict[1], player_dict[88])

ComputeDistance(player_dict[1], player_dict[76])

display(ComputeDistance(player_dict[1], player_dict[20]))  ####Wentz

display(ComputeDistance(player_dict[1], player_dict[21]))  ###Wilson

display(ComputeDistance(player_dict[1], player_dict[71]))  ##Newton

display(ComputeDistance(player_dict[1], player_dict[5]))   ###Mahomes

display(ComputeDistance(player_dict[1], player_dict[3]))   ###Prescott

ComputeDistance(player_dict[1], player_dict[79])

ComputeDistance(player_dict[1], player_dict[58])      ####Ryan
getNeighbors(1, 90)

ComputeDistance(player_dict[1], player_dict[2])  ###AJ McCarron