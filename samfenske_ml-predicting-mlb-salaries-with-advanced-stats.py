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
hitting_2014=pd.read_csv('/kaggle/input/hitting/adjusted_2014_hitting').drop(columns='Unnamed: 0')

pd.set_option('display.max_columns', None)

hitting_2014
#plate appearances

walks=hitting_2014['bb']+hitting_2014['hbp']

sacrifice_hits=hitting_2014['sf']+hitting_2014['sh']

hitting_2014['PA']=hitting_2014['ab']+walks+sacrifice_hits
#total bases

singles=hitting_2014['h']-(hitting_2014['double']+hitting_2014['triple']+hitting_2014['hr'])

doubles=hitting_2014['double']

triples=hitting_2014['triple']

hr=hitting_2014['hr']

hitting_2014['tb']=singles + 2*doubles + 3*triples + 4*hr

hitting_2014
#batting average

hitting_2014['avg']=hitting_2014['h']/hitting_2014['ab']



#on base percentage

hitting_2014['OBP']=(hitting_2014['h']+walks)/(hitting_2014['ab']+walks+hitting_2014['sf'])



#slugging percentage

hitting_2014['SLG']=hitting_2014['tb']/hitting_2014['ab']



#isolated slugging percentage

hitting_2014['ISO']=hitting_2014['SLG']-hitting_2014['OBP']



hitting_2014
hitting_2014.sort_values(by=['ISO','avg'],ascending=False).head(20)
import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(20,10))

sns.heatmap(hitting_2014.corr(),annot=True,linewidth=0.5)
salaries=pd.read_csv('/kaggle/input/the-history-of-baseball/salary.csv')

recent=salaries[salaries['year'].isin(['2015'])]



batting=pd.read_csv('/kaggle/input/the-history-of-baseball/batting.csv')

hitting=batting[batting['year'].isin([2012,2013,2014])]

players=recent['player_id'].tolist()

hitting_filtered=hitting[hitting['player_id'].isin(players)]

hitting_filtered
len(hitting_filtered['year'].value_counts())
players=hitting_filtered['player_id'].value_counts().index

adjusted_hitting=pd.DataFrame()

for player in players:

    player_df=hitting[hitting['player_id'].isin([player])]

    numeric_stats=['g','ab','r','h','double','triple','hr','rbi','sb','cs','bb','so','ibb','hbp','sh','sf','g_idp']

    

    traded=pd.DataFrame()

    not_traded=pd.DataFrame()

    #if player was ever traded

    if player_df['year'].value_counts().reset_index()['year'][0]>1:

        traded=traded.append({'player_id':player,'years':len(player_df['year'].value_counts()),'stint':len(player_df),

                'team_id':'trade','league_id':'trade'},ignore_index=True)

        for stat in numeric_stats:

            traded[stat]=player_df.sum()[stat]

        adjusted_hitting=adjusted_hitting.append(traded)

    else:

        not_traded=not_traded.append({'player_id':player,'years':len(player_df),'stint':len(player_df),'team_id':player_df.reset_index()['team_id'][0],

                                      'league_id':player_df.reset_index()['league_id'][0]},ignore_index=True)

        for stat in numeric_stats:

            not_traded[stat]=player_df.sum()[stat]

        adjusted_hitting=adjusted_hitting.append(not_traded)

adjusted_hitting
adjusted_hitting=adjusted_hitting.reset_index().drop(columns='index')
adjusted_hitting['salary']=[recent[recent['player_id'].isin([player])].reset_index()

                        ['salary'][0] for player in adjusted_hitting['player_id']]

adjusted_hitting
fielding=pd.read_csv('/kaggle/input/the-history-of-baseball/fielding.csv')

fielding=fielding[fielding['player_id'].isin(players)&fielding['year'].isin([2012,2013,2014])]

positions_adjusted=pd.DataFrame()

for player in players:

    player_df=fielding[fielding['player_id'].isin([player])]

    positions=player_df['pos'].value_counts().index

    tracker=pd.DataFrame()

    for position in positions:

        df=player_df[player_df['pos'].isin([position])]

        tracker=tracker.append({'pos':position,'games':df.sum()['g']},ignore_index=True)

    id_max=tracker['games'].idxmax()

    positions_adjusted=positions_adjusted.append({'player_id':player,'pos':tracker['pos'][id_max]},ignore_index=True)

adjusted_hitting['pos']=[positions_adjusted[positions_adjusted['player_id'].isin([player])].reset_index()

                        ['pos'][0] for player in adjusted_hitting['player_id']]

adjusted_hitting=adjusted_hitting[-adjusted_hitting['pos'].isin(['P'])].reset_index().drop(columns='index')
names=pd.read_csv('/kaggle/input/the-history-of-baseball/player.csv')

names['name']=names['name_first']+' '+names['name_last']

names=names[['player_id','name']]

adjusted_hitting['name']=[names[names['player_id'].isin([player])].reset_index()

                        ['name'][0] for player in adjusted_hitting['player_id']]
#plate appearances

walks=adjusted_hitting['bb']+adjusted_hitting['hbp']

sacrifice_hits=adjusted_hitting['sf']+adjusted_hitting['sh']

adjusted_hitting['PA']=adjusted_hitting['ab']+walks+sacrifice_hits



#total bases

singles=adjusted_hitting['h']-(adjusted_hitting['double']+adjusted_hitting['triple']+adjusted_hitting['hr'])

doubles=adjusted_hitting['double']

triples=adjusted_hitting['triple']

hr=adjusted_hitting['hr']

adjusted_hitting['tb']=singles + 2*doubles + 3*triples + 4*hr



#batting average

adjusted_hitting['avg']=adjusted_hitting['h']/adjusted_hitting['ab']



#on base percentage

adjusted_hitting['OBP']=(adjusted_hitting['h']+walks)/(adjusted_hitting['ab']+walks+adjusted_hitting['sf'])



#slugging percentage

adjusted_hitting['SLG']=adjusted_hitting['tb']/adjusted_hitting['ab']



#isolated slugging percentage

adjusted_hitting['ISO']=adjusted_hitting['SLG']-adjusted_hitting['OBP']

adjusted_hitting
plt.figure(figsize=(20,10))

sns.heatmap(adjusted_hitting.corr(),annot=True,linewidth=0.5)
df=pd.DataFrame(adjusted_hitting.corr()['salary']).reset_index()

df['Beat Threshold']=abs(df['salary'])>0.5



sns.lmplot(x='index', y="salary", data=df,hue='Beat Threshold',fit_reg=False,height=4,

           aspect=4).set_xticklabels(rotation=90)
adjusted_hitting.corr().nlargest(6,columns='salary')['salary']
hitting_2014.corr().nlargest(6,columns='salary')['salary']
features=['rbi','tb','r','h','double']

y=adjusted_hitting['salary']

X=adjusted_hitting[features]



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

basic_model = DecisionTreeRegressor(random_state=1)

basic_model.fit(train_X, train_y)

predictions=basic_model.predict(val_X)



df=pd.DataFrame(val_X)

df['prediction']=predictions

df['ID']=[adjusted_hitting['player_id'][index] for index in df.reset_index()['index']]

df['name']=[adjusted_hitting['name'][index] for index in df.reset_index()['index']]

df['pos']=[adjusted_hitting['pos'][index] for index in df.reset_index()['index']]

df['salary']=[adjusted_hitting['salary'][index] for index in df.reset_index()['index']]

df=df[['name','ID','r','h','double','rbi','tb','pos','salary','prediction']]

df['excess']=df['prediction']-df['salary']
from sklearn.metrics import mean_absolute_error

mean_absolute_error(df['salary'], df['prediction'])
df['absolute error']=abs(df['excess'])

df.sort_values(by='absolute error').head(20)
fa=pd.read_excel('/kaggle/input/freeagents/FA.xlsx')

fa
fa_stats=adjusted_hitting[adjusted_hitting['name'].isin(fa['PLAYER'].tolist())]

fa_stats
features=['rbi','tb','r','h','double']

y=fa_stats['salary']

X=fa_stats[features]



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state=1, test_size=0.4)

basic_model = DecisionTreeRegressor(random_state=1)

basic_model.fit(train_X, train_y)

predictions=basic_model.predict(val_X)



df=pd.DataFrame(val_X)

df['prediction']=predictions

df['ID']=[adjusted_hitting['player_id'][index] for index in df.reset_index()['index']]

df['name']=[adjusted_hitting['name'][index] for index in df.reset_index()['index']]

df['pos']=[adjusted_hitting['pos'][index] for index in df.reset_index()['index']]

df['salary']=[adjusted_hitting['salary'][index] for index in df.reset_index()['index']]

df=df[['name','ID','r','h','double','rbi','tb','pos','salary','prediction']]

df['excess']=df['prediction']-df['salary']
from sklearn.metrics import mean_absolute_error

mean_absolute_error(df['salary'], df['prediction'])