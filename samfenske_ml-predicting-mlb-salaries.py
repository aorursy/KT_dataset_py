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
salaries=pd.read_csv('/kaggle/input/baseball-databank/Salaries.csv')

salaries
recent=salaries[salaries['yearID'].isin(['2015'])]

recent
recent.sort_values(by='salary',ascending=False)
#df of all player IDs mapping to player names

names=pd.read_csv('/kaggle/input/the-history-of-baseball/player.csv')

pd.set_option('display.max_columns', None)



#make column combining first and last name

names['name']=names['name_first']+' '+names['name_last']



#only need these two columns (rest of columns are unnecessary)

names=names[['player_id','name']]

names
batting=pd.read_csv('/kaggle/input/the-history-of-baseball/batting.csv')



#get batting data from 2014

hitting_2014=batting[batting['year'].isin(['2014'])]

hitting_2014
#all player IDs from 2014 salaries dataframe, need to convert to list for next step

players_2014=recent['playerID'].tolist()



#hitting data for players in 2014 salaries dataframe

hitting_2014_filtered=hitting_2014[hitting_2014['player_id'].isin(players_2014)]

hitting_2014_filtered
recent['playerID'].value_counts()
hitting_2014_filtered['player_id'].value_counts()
players=hitting_2014_filtered['player_id'].value_counts().index

adjusted_2014_hitting=pd.DataFrame()

for player in players:

    player_df=hitting_2014_filtered[hitting_2014_filtered['player_id'].isin([player])]

    if len(player_df)>1:

        numeric_stats=['g','ab','r','h','double','triple','hr','rbi','sb','cs','bb','so','ibb','hbp','sh','sf','g_idp']

        

        df=pd.DataFrame()

#         df=player_df.sum()[numeric_stats]

        df=df.append({'player_id':player,'year':'2014','stint':len(player_df),

                'team_id':'trade','league_id':'trade'},ignore_index=True)

        for stat in numeric_stats:

            df[stat]=player_df.sum()[stat]

        adjusted_2014_hitting=adjusted_2014_hitting.append(df)

    else:

        adjusted_2014_hitting=adjusted_2014_hitting.append(player_df)

adjusted_2014_hitting
adjusted_2014_hitting=adjusted_2014_hitting.reset_index().drop(columns='index')
recent
adjusted_2014_hitting['salary']=[recent[recent['playerID'].isin([player])].reset_index()

                        ['salary'][0] for player in adjusted_2014_hitting['player_id']]
adjusted_2014_hitting.sort_values(by='salary',ascending=False)
fielding=pd.read_csv('/kaggle/input/the-history-of-baseball/fielding.csv')

fielding_2014=fielding[fielding['player_id'].isin(players_2014)&fielding['year'].isin(['2014'])]

fielding_2014['player_id'].value_counts()
fielding_2014[fielding_2014['player_id'].isin(['johnske05'])]
positions_adjusted=pd.DataFrame()

for player in players:

    player_df=fielding_2014[fielding_2014['player_id'].isin([player])]

    positions=player_df['pos'].value_counts().index

    tracker=pd.DataFrame()

    for position in positions:

        df=player_df[player_df['pos'].isin([position])]

        tracker=tracker.append({'pos':position,'games':df.sum()['g']},ignore_index=True)

    id_max=tracker['games'].idxmax()

    positions_adjusted=positions_adjusted.append({'player_id':player,'pos':tracker['pos'][id_max]},ignore_index=True)

positions_adjusted
adjusted_2014_hitting['pos']=[positions_adjusted[positions_adjusted['player_id'].isin([player])].reset_index()

                        ['pos'][0] for player in adjusted_2014_hitting['player_id']]

adjusted_2014_hitting
adjusted_2014_hitting=adjusted_2014_hitting[-adjusted_2014_hitting['pos'].isin(['P'])].reset_index().drop(columns='index')

adjusted_2014_hitting
adjusted_2014_hitting['name']=[names[names['player_id'].isin([player])].reset_index()

                        ['name'][0] for player in adjusted_2014_hitting['player_id']]

adjusted_2014_hitting
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(20,10))

sns.heatmap(adjusted_2014_hitting.corr(),annot=True,linewidth=0.5)
df=pd.DataFrame(adjusted_2014_hitting.corr()['salary']).reset_index()

df['Beat Threshold']=abs(df['salary'])>0.5



sns.lmplot(x='index', y="salary", data=df,hue='Beat Threshold',fit_reg=False,height=4,

           aspect=4).set_xticklabels(rotation=90)
features=['ab','r','h','double','rbi','bb','sf']
def scatter(attribute):

    p1=sns.lmplot(x=attribute, y="salary", data=adjusted_2014_hitting,fit_reg=False,height=8,aspect=4)

    ax = p1.axes[0,0]

    for i in range(len(adjusted_2014_hitting)):

        ax.text(adjusted_2014_hitting[attribute][i], adjusted_2014_hitting['salary'][i], adjusted_2014_hitting['name'][i],

               fontsize='small',rotation=45)
scatter('rbi')
scatter('bb')
y=adjusted_2014_hitting['salary']

X=adjusted_2014_hitting[features]



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

basic_model = DecisionTreeRegressor(random_state=1)

basic_model.fit(train_X, train_y)

predictions=basic_model.predict(val_X)
df=pd.DataFrame(val_X)

df['prediction']=predictions

df['ID']=[adjusted_2014_hitting['player_id'][index] for index in df.reset_index()['index']]

df['name']=[adjusted_2014_hitting['name'][index] for index in df.reset_index()['index']]

df['pos']=[adjusted_2014_hitting['pos'][index] for index in df.reset_index()['index']]

df['salary']=[adjusted_2014_hitting['salary'][index] for index in df.reset_index()['index']]

df=df[['name','ID','ab','r','h','double','rbi','bb','pos','salary','prediction']]

df
df['excess']=df['prediction']-df['salary']

df.sort_values(by='excess')
df.style.format({'prediction': "{0:,.2f}",'salary': "{0:,.2f}",'excess': "{0:,.2f}"})

df=df.astype({"ab": int,"r": int,"h": int,"double": int,"rbi": int,"bb":int}) 

df
type(df.style.format({'prediction': "{0:,.2f}",'salary': "{0:,.2f}",'excess': "{0:,.2f}"})

)
abs(df['excess']).mean()
from sklearn.metrics import mean_absolute_error

mean_absolute_error(df['salary'], df['prediction'])
df['salary'].mean()
df['salary'].median()
adjusted_2014_hitting.to_csv('ML1')