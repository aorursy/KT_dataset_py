# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
wnba = pd.read_csv("../input/wnba-player-stats-2017/WNBA Stats.csv")
wnba.head()
wnba.tail()
wnba.columns
parameter = wnba['Games Played'].max()

sample= wnba['Games Played'].sample(30, random_state=1)

statistic= sample.max()

sampling_error= parameter- statistic



sampling_error
import matplotlib.pyplot as plt

sample_means=[]

pts_mean=wnba['PTS'].mean()

for i in range(100):

    sample=wnba['PTS'].sample(10,random_state=i)

    sample_means.append(sample.mean())

   

    

plt.ylabel('Sample Mean')

plt.xlabel('Sample Number')

plt.scatter(range(1,101), sample_means)

plt.axhline(pts_mean)
wnba['Pts_per_game']= wnba['PTS'] /wnba['Games Played']

# Stratifying the data in five strata

stratum_G =wnba[wnba.Pos =='G']

stratum_F =wnba[wnba.Pos =='F']

stratum_C =wnba[wnba.Pos =="C"]

stratum_GF =wnba[wnba.Pos =="G/F"]

stratum_FC= wnba[wnba.Pos=="F/C"]

points_position = {}

for startum, position in [(stratum_G, 'G'),(stratum_F, 'F'),

                          (stratum_C,'C'),(stratum_GF, 'G/F'),

                          (stratum_FC, 'F/C')

                         ]:

    sample=startum['Pts_per_game'].sample(10,random_state=0)

    points_position[position]=sample.mean()

    

position_most_points = max(points_position, key=points_position.get)    

position_most_points
wnba['Pts_per_game']
print(wnba['Games Played'].min())
print(wnba['Games Played'].max())
print(wnba['Games Played'].value_counts(bins = 3, normalize = True) * 100)
under_12 = wnba[wnba['Games Played'] <=12]

bet13to22 = wnba[(wnba['Games Played']> 12) & (wnba['Games Played']<=22)]

over_23= wnba[wnba['Games Played']> 22]

proportional_sampling_means = []

for i in range(100):

    sample_under_12=under_12['PTS'].sample(1, random_state=i)

    sample_btw_13_22 =bet13to22['PTS'].sample(2, random_state=i)

    sample_over_23=over_23["PTS"].sample(7,random_state=i)

    

    final_sample = pd.concat([sample_under_12, sample_btw_13_22, sample_over_23])

    proportional_sampling_means.append(final_sample.mean()) 

              

plt.scatter(range(1,101), proportional_sampling_means)

plt.axhline(wnba['PTS'].mean()) 
wnba['MIN'].min()
wnba['MIN'].max()
(wnba['MIN'].value_counts(bins = 5, normalize = True)*100)
(wnba['MIN'].value_counts(bins = 3, normalize = True)*100)

under_min= wnba[wnba['MIN'] <= 347.333]

bet_min=wnba[(wnba['MIN'] > 347.333) & (wnba['MIN'] <=682.667)]

over_min=wnba[wnba['MIN'] >682.667]



points_per_proportion=[]

for i in range(100):

    sample_under_min = under_min['PTS'].sample(4, random_state=i)

    sample_bet_min = bet_min['PTS'].sample(4, random_state=i)

    sample_over_min= over_min['PTS'].sample(4,random_state=i)

    final_sample= pd.concat([sample_under_min,sample_bet_min,sample_over_min])

    points_per_proportion.append(final_sample.mean())

    

plt.scatter(range(1,101),points_per_proportion) 

plt.axhline(wnba['PTS'].mean())
(wnba['MIN'].value_counts(bins = 5, normalize = True)*100)

undermins= wnba[(wnba['MIN'] >= 10.993) & (wnba['MIN'] <= 127)]

under_min= wnba[(wnba['MIN'] >= 127) & (wnba['MIN'] <= 263)]

under_min1= wnba[(wnba['MIN'] >= 263) & (wnba['MIN'] <= 515)]

bet_min=wnba[(wnba['MIN'] > 515) & (wnba['MIN'] <=766.5)]

over_min=wnba[wnba['MIN'] >766.5]



points_per_proportion=[]

for i in range(100):

    sample_undermins = undermins["PTS"].sample(1, random_state=i)

    sample_under_min = under_min['PTS'].sample(3, random_state=i)

    sample_under_min1 = under_min1['PTS'].sample(4, random_state=i)

    sample_bet_min = bet_min['PTS'].sample(4, random_state=i)

    sample_over_min= over_min['PTS'].sample(3,random_state=i)

    final_sample= pd.concat([sample_under_min, sample_under_min1,sample_bet_min,sample_over_min])

    points_per_proportion.append(final_sample.mean())

    

plt.scatter(range(1,101),points_per_proportion) 

plt.axhline(wnba['PTS'].mean())