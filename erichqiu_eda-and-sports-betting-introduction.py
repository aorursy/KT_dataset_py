import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
files = [filename for filename in os.listdir('/kaggle/input/nba-odds-and-scores') if os.path.isdir(os.path.join('/kaggle/input/nba-odds-and-scores',filename))]

odds = []

scores = []

for file in files:

    temp_odds = pd.read_csv('../input/nba-odds-and-scores/'+file+'/vegas.txt',index_col=0,parse_dates=True)

    temp_scores = pd.read_csv('../input/nba-odds-and-scores/'+file+'/raw_scores.txt',index_col=0,parse_dates=True)

    odds.append(temp_odds)

    scores.append(temp_scores)

master_odds = pd.concat(odds,axis=0)

master_scores = pd.concat(scores,axis=0)
spreads = master_odds[master_odds['Location']=='away']['Average_Line_Spread']

spreads.hist()

plt.title('Histogram of Away Team Spreads from 2012-2019')

plt.xlabel('Spread')

plt.show()

avg_spread = spreads.mean()

print('Average Spread is '+str(np.round(avg_spread,2))+' and this is how much home court is worth.')
ou = master_odds[master_odds['Location']=='away']['Average_Line_OU']

ou.hist()

plt.title('Histogram of Road O/U from 2012-2019')

plt.xlabel('Spread')

plt.show()

avg_ou = ou.mean()

print('Average O/U is '+str(np.round(avg_ou,2))+' and this is how much teams typically score during this time period.')

temp_data = master_odds[master_odds['Location']=='away'][['Average_Line_OU']]

temp_data['year'] = temp_data.index.year

sns.boxplot(x='year',y='Average_Line_OU',data=temp_data)

plt.title('Box Plot of Year by Year O/U')

plt.show()
ml = master_odds[master_odds['Location']=='away']['Average_Line_ML']

ml.hist()

plt.title('Histogram of Road ML from 2012-2019')

plt.xlabel('ML')

plt.show()

avg_ml = ml.mean()

print('Average ML is '+str(np.round(avg_ml,2))+' for road teams, meaning according to the implied odds they have around a 44% chance of winning.')
regression_frame = pd.concat([spreads,ou],axis=1)

regression_frame.columns = ['Spread','OU']

regression_frame.describe().T
def classification_stats(x):

    return pd.Series([np.round(x.sum()/len(x),3),int(x.sum()),int(len(x))],index=['Percent','Counts','Total'])



road_cover = master_odds[master_odds['Location']=='away']['Spread']>master_odds[master_odds['Location']=='away']['Average_Line_Spread']*-1

over = master_odds[master_odds['Location']=='away']['Total']>master_odds[master_odds['Location']=='away']['Average_Line_OU']

road_win = master_odds[master_odds['Location']=='away']['Result']=='W'

classification_frame = pd.concat([road_cover,over,road_win],axis=1)

classification_frame.columns = ['Road Cover','Over Cover','Road Win']

classification_frame.apply(classification_stats).T