import numpy as np

import pandas as pd



import os

import matplotlib.pyplot as plt

import seaborn as sns



pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 3000)

pd.set_option('display.width', 3000)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Per player, per season, per statistic

year_averages = pd.read_csv('/kaggle/input/pga-tour-20102018-data/PGA_Data_Historical.csv')





# Weekly numbers - consolidate 2 datasets into 1

data2020 = pd.read_csv('/kaggle/input/pga-tour-20102018-data/2020_data.csv')

data2019 = pd.read_csv('/kaggle/input/pga-tour-20102018-data/2019_data.csv')

data = pd.concat([data2019, data2020])

data = data[~data['player_name'].isna()]

data.head()
#data['statistic'].unique()
player_cols = [#'Top 10 Finishes', 

               #'Club Head Speed', 'Ball Speed', 'Smash Factor', 'Launch Angle', 'Spin Rate',

       #'Distance to Apex', 'Apex Height', 'Hang Time', 'Carry Distance',

       #'Carry Efficiency', 'Power Rating',

       #'Accuracy Rating', 'Short Game Rating',

       #'Last 5 Events - Putting', 'Last 5 Events - Scoring',

       #'Last 15 Events - Putting', 'Last 15 Events - Scoring',

              'SG: Off-the-Tee',

    'SG: Approach the Green',

       'SG: Total',

       'SG: Around-the-Green',

       'SG: Putting', 

 'SG: Tee-to-Green',

    'Percentage of potential money won',

'Official Money',

'Stroke Differential Field Average',

'Scoring Average']

# Revised dataset

data = data[data['statistic'].isin(player_cols)]

data = data[data['variable'] != 'RANK THIS WEEK']

data = data.drop(columns = ['variable'])

data.head()
data.head()
data = data.set_index(['player_name', 'date', 'tournament','statistic'])['value'].unstack('statistic').reset_index()

data.head()
# Approach, around green, off the tee

data.rename(columns = {'SG: Approach the Green':'Approach',

                       'SG: Around-the-Green':'Around_green',

                       'SG: Off-the-Tee':'Tee',

                      'SG: Putting':'Putting',

                      'SG: Tee-to-Green':'Tee2Green',

                      'SG: Total':'Total',

                      'Official Money': 'Money',

                      'Percentage of potential money won' : 'Money Pct',

                      "Stroke Differential Field Average": 'Better than Avg'}, inplace = True)

data.head()
wide = 7; tall = 7

fig = plt.figure(figsize = [wide, tall])

fig.suptitle("Avg Strokes Gained", fontsize=14)





ax1 = fig.add_subplot(2,2,1)

sns.distplot(data["Approach"], 

                hist = False, 

                kde = True, 

                kde_kws = {'shade':True}, 

                rug = False, 

                bins = 10,

                hist_kws = dict(alpha=1))

ax1.set_title("Approach")

ax1.set_xlabel("")

ax1.set_xticks(range(-4,5,2)) 

ax1.set_xticklabels(range(-4,5,2), fontsize=12)

ax1.set_ylabel("")







ax2 = fig.add_subplot(2,2,2)

sns.distplot(data["Around_green"], 

                hist = False, 

                kde = True, 

                kde_kws = {'shade':True}, 

                rug = False, 

                bins = 10,

                hist_kws = dict(alpha=1))

ax2.set_title("Around_green")

ax2.set_xlabel("")

ax2.set_xticks(range(-4,5,2)) 

ax2.set_xticklabels(range(-4,5,2), fontsize=12)

ax2.set_ylabel("")





ax3 = fig.add_subplot(2,2,3)

sns.distplot(data["Tee"], 

                hist = False, 

                kde = True, 

                kde_kws = {'shade':True}, 

                rug = False, 

                bins = 10,

                hist_kws = dict(alpha=1))

ax3.set_title("Tee")

ax3.set_xlabel("")

ax3.set_xticks(range(-4,5,2)) 

ax3.set_xticklabels(range(-4,5,2), fontsize=12)

ax3.set_ylabel("")





ax4 = fig.add_subplot(2,2,4)

sns.distplot(data["Putting"], 

                hist = False, 

                kde = True, 

                kde_kws = {'shade':True}, 

                rug = False, 

                bins = 10,

                hist_kws = dict(alpha=1))

ax4.set_title("Putting")

ax4.set_xlabel("")

ax4.set_xticks(range(-4,5,2)) 

ax4.set_xticklabels(range(-4,5,2), fontsize=12)

ax4.set_ylabel("")



fig.subplots_adjust(hspace=0.5, wspace=0.3)
data.to_csv('golf.csv',index=False)