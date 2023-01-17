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
regular_detail = pd.read_csv('../input/march-madness-analytics-2020/MDataFiles_Stage2/MRegularSeasonDetailedResults.csv')

tour_detail = pd.read_csv('../input/march-madness-analytics-2020/MDataFiles_Stage2/MNCAATourneyDetailedResults.csv')

# create new columns for 2 points shot

regular_detail['WFGM2'] = regular_detail['WFGM'] - regular_detail['WFGM3']

regular_detail['WFGA2'] = regular_detail['WFGA'] - regular_detail['WFGA3']

regular_detail['LFGM2'] = regular_detail['LFGM'] - regular_detail['LFGM3']

regular_detail['LFGA2'] = regular_detail['LFGA'] - regular_detail['LFGA3']



w_detail = regular_detail[['Season', 'WScore', 'WFGM', 'WFGA', 'WFGM2', 'WFGA2', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA',

                                'WOR', 'WDR', 'WAst','WTO', 'WStl', 'WBlk', 'WPF']]

l_detail = regular_detail[['Season', 'LScore', 'LFGM', 'LFGA', 'LFGM2', 'LFGA2', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',

                                'LOR', 'LDR', 'LAst','LTO', 'LStl', 'LBlk', 'LPF']]



new_columns = ['Season', 'Score', 'FGM', 'FGA', 'FGM2', 'FGA2', 'FGM3', 'FGA3', 'FTM', 

               'FTA', 'OR', 'DR', 'Ast','TO', 'Stl', 'Blk', 'PF']





w_detail.columns = new_columns

l_detail.columns = new_columns



all_detail = pd.concat([w_detail, l_detail])



w_detail = w_detail.groupby('Season').mean().round(2)

l_detail = l_detail.groupby('Season').mean().round(2)

all_detail = all_detail.groupby('Season').mean().round(2)

all_detail
w_detail
l_detail
import matplotlib.pyplot as plt

import seaborn as sns

sns.set();

%matplotlib inline

plt.rcParams["figure.figsize"] = (10,5)



year = [i for i in range(2003, 2021)]
plt.plot(year, w_detail.Score, marker='o')

plt.plot(year, l_detail.Score, marker='o')

plt.xticks(year);



plt.title('Average Score of Winning and Losing Teams Over Time');

fig, (ax, ax2) = plt.subplots(2, 1)



fig.suptitle('3-Point Attempt VS 2-Point Attempt');

ax.plot(year, all_detail.FGA2, marker='o', label='2-Point', c='r')

ax.set_yticks([i for i in range(35, 40)])

ax.set_xticks(year);



ax2.plot(year, all_detail.FGA3, marker='o', label='3-Point')

ax2.set_xticks(year);



for ax in fig.get_axes():

    ax.label_outer()
plt.plot(year, all_detail.FGM3/all_detail.FGA3, marker='o')

plt.xticks(year);

plt.title('3-Point Percentage Change');