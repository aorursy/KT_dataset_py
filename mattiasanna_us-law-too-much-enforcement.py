# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



#data visualisation

import seaborn as sns 

from matplotlib import pyplot as plt 

import plotly.graph_objects as go

import plotly.figure_factory as ff



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/us-police-shootings/shootings.csv')

df.info()
df.head()
from plotly.subplots import make_subplots

colors = ['indianred','crimson']

fig = go.Figure()

fig.add_trace(go.Bar(x = df['manner_of_death'].unique(), y = df['manner_of_death'].value_counts(), marker_color = colors, name ='manner of death'))

fig.update_layout(boxmode='group',width = 800)

fig.show()
data = df['threat_level'].value_counts()

fig = go.Figure()

fig.add_trace(go.Bar(x = data.index, y = data, marker_color = 'indianred', name ='Threat level'))

fig.update_layout(width = 800)

fig.show()
data = df.groupby(['threat_level','armed'])['race'].value_counts()

fig = go.Figure()

fig.add_trace(go.Bar(x=data.loc['other']['unarmed'].index, y = data.loc['other']['unarmed'], marker_color = 'indianred'))

fig.update_layout(title = 'Unarmed and not attacking deaths per Race')

fig.show()
col = ['manner_of_death','threat_level']

data = df.groupby(col)['arms_category'].value_counts()

print('Percentage of people shot and Tasered not considered attacking {}% \n'.format(round(data.loc['shot and Tasered']['other'].sum()*100/248,1)))

print('Weapon distribution of People not considered attacking, that have been shot and Tasered \n\n {}'.format(data.loc['shot and Tasered']['other']))

df['age'] = df['age'].astype(int)

df['age'].describe()
data = df.groupby('gender')['age'].value_counts()

fig = make_subplots(rows =1, cols =2,specs=[[{"type": "box"},{"type": "pie"}]], column_titles = ['Age Distribution per Gender', 'Gender Count and Ratio'])

fig.add_trace(go.Box(x = [i[0] for i in data.index], y = [j[1] for j in data.index], showlegend = False, boxmean = True, name = 'gender age'),1,1)

fig.add_trace(go.Pie(labels= df['gender'].unique(), values = df['gender'].value_counts(),showlegend = True, name ='quantities'), row =1, col = 2)

fig.update_layout(boxmode='group', width = 800)

fig.show()
data = df.groupby('race')['gender'].value_counts()

print('Age distribution per Race \n \n{}'.format(data))
data = df.groupby(['race'])['age'].value_counts()

fig = go.Figure()

fig.add_trace(go.Box(x = [i[0] for i in data.index], y =[j[1] for j in data.index], marker_color = 'indianred', boxmean = True))

fig.update_layout(yaxis = dict(title = 'Deaths count'))

fig.show()
fig = go.Figure()

for race in df['race'].unique():

    fig.add_trace(go.Box(y =df['age'][df['race'] == race], marker_color = 'indianred',name = race, boxmean = True))

    fig.update_layout(yaxis = dict(title = 'Victims count'), showlegend = False)



fig.show()
fig = go.Figure()

fig.add_trace(go.Pie(labels= df['race'].value_counts().index, values = df['race'].value_counts()))

fig.show()
df['date'] = pd.to_datetime(df['date'])

df['Year'] = df['date'].dt.year

df['Month'] = df['date'].dt.strftime('%b')

df['Week Day']= df['date'].dt.day_name()



race_2017 = df['race'][df['Year'] == 2017].value_counts()[:-1]

race_2016 = df['race'][df['Year'] == 2016].value_counts()[:-1]

race_2015 = df['race'][df['Year'] == 2015].value_counts()

race_2015.pop('Other')

race_USA_2016 = [197.479, 39.717, 57.398, 2.676+0.595, 17.556]

race_USA_2015 = [197.534, 39.597, 56.496, 17.273, 2.597+0.554]

race_USA_2017 = [197.285, 40.129, 58.846,2.726+0.608, 18.215,]

proportion_2017 = round(race_2017/race_USA_2017,3)

proportion_2016 = round(race_2016/race_USA_2016,3)

proportion_2015 = round(race_2015/race_USA_2015,3)

colors = ['darkred','crimson', 'firebrick', 'sienna','peru']

fig = go.Figure()

fig.add_trace(go.Bar(x = proportion_2015.index, y = proportion_2015, name = '2015' ))

fig.add_trace(go.Bar(x = proportion_2016.index, y = proportion_2016, name = '2016' ))

fig.add_trace(go.Bar(x = proportion_2017.index, y = proportion_2017, name = '2017' ))

fig.update_layout(barmode='group',title_text = 'Deaths per Race per millions people in 2015-2017')

fig.show()
df_pop = pd.read_csv('../input/stateus-pop-and-race-dist/statesUS - states-Foglio2.csv')

for i,line in enumerate(df_pop['Population']):

    df_pop['Population'].loc[i] = line.replace(",","")

df_pop['Population'] = df_pop['Population'].astype(int)

df_pop.head()
state_2019 = df['state'][df['Year']==2019].value_counts()

prop_states_2019 = []

for i,state in enumerate(df_pop['Abbreviation']):

    try: prop_states_2019.append((state,state_2019[state]*1000000/df_pop['Population'].loc[i]))

    except KeyError: continue

fig = go.Figure()

fig.add_trace(go.Bar(x = [i[0] for i in prop_states_2019], y =[round(j[1],3) for j in prop_states_2019], marker = dict(color = [round(j[1],3) for j in prop_states_2019])))

fig.update_layout(title_text = 'Deaths per State per millions Inhabitants 2019')

fig.show()
region = {'CT':'NE','DE':'NE','DC':'NE','ME':'NE','MD':'NE','MA':'NE','NH':'NE','NJ':'NE','NY':'NE','PA':'NE','RI':'NE','VT':'NE',

          'ND':'MW','SD':'MW','NE':'MW','KS':'MW','MO':'MW','IA':'MW','MN':'MW','WI':'MW','MI':'MW','IL':'MW','IN':'MW','OH':'MW',

          'AL':'S','AR':'S','FL':'S','GA':'S','KY':'S','LA':'S','MS':'S','NC':'S','SC':'S','TN':'S','VA':'S','WV':'S','TX':'S','OK':'S',

          'CO':'W','ID':'W','MT':'W','NV':'W','UT':'W','WY':'W','AK':'W','CA':'W','HI':'W','OR':'W','WA':'W','AZ':'W','NM':'W','WY':'W'}

df['region'] = df['state'].map(region)

#df.loc[df['region'].isna()]['state'].unique()
tab = pd.crosstab(df['race'],df['region'], margins = False)

tab
n_tab = pd.concat([tab.iloc[1],tab.iloc[2],tab.iloc[5]],axis = 1)



NE_race_dist = [6.55,6.74,37.9] #Black, Hispanic, White

MW_race_dist = [6.92,4.67,52] 

S_race_dist = [22.3,18.4,69] 

W_race_dist = [3.41,20.8,38.1] 



n_tab.iloc[0] = n_tab.iloc[0]/MW_race_dist

n_tab.iloc[1] = n_tab.iloc[1]/NE_race_dist

n_tab.iloc[2] = n_tab.iloc[2]/S_race_dist

n_tab.iloc[3] = n_tab.iloc[3]/W_race_dist

n_tab = round(n_tab).astype(int)

n_tab
from scipy.stats import chi2_contingency

stat,p,dof,expected = chi2_contingency(n_tab)

p,expected
exp_H = np.array([(tab['MW'].sum()/sum(MW_race_dist))*MW_race_dist[1],(tab['NE'].sum()/sum(NE_race_dist))*NE_race_dist[1], (tab['S'].sum()/sum(S_race_dist))*S_race_dist[1],(tab['W'].sum()/sum(W_race_dist))*W_race_dist[1]])

exp_B = np.array([(tab['MW'].sum()/sum(MW_race_dist))*MW_race_dist[0],(tab['NE'].sum()/sum(NE_race_dist))*NE_race_dist[0], (tab['S'].sum()/sum(S_race_dist))*S_race_dist[0],(tab['W'].sum()/sum(W_race_dist))*W_race_dist[0]])

exp_W = np.array([(tab['MW'].sum()/sum(MW_race_dist))*MW_race_dist[2],(tab['NE'].sum()/sum(NE_race_dist))*NE_race_dist[2], (tab['S'].sum()/sum(S_race_dist))*S_race_dist[2],(tab['W'].sum()/sum(W_race_dist))*W_race_dist[2]])
fig = make_subplots(rows = 1, cols = 3)

fig.add_trace(go.Bar(x = tab.columns, y=tab.loc['Black'], name= 'Observed Black Deaths'),1,1)

fig.add_trace(go.Bar(x = tab.columns, y=exp_B,name = 'Expected Black Deaths'),1,1)



fig.add_trace(go.Bar(x = tab.columns, y=exp_H, name = 'Expected Hispanic Deaths'),1,2)

fig.add_trace(go.Bar(x = tab.columns, y=tab.loc['Hispanic'], name= 'Observed Hispanic Deaths'),1,2)



fig.add_trace(go.Bar(x = tab.columns, y=exp_W, name = 'Expected White Deaths'),1,3)

fig.add_trace(go.Bar(x = tab.columns, y=tab.loc['White'], name= 'Observed White Deaths'),1,3)



fig.update_traces(opacity = 0.6)

fig.update_layout(barmode='overlay',title_text = 'Death per Region: Obsereved vs Expected',showlegend = True)

fig.show()
obs_B = np.array(tab.loc['Black'])

obs_H = np.array(tab.loc['Hispanic'])

obs_W = np.array(tab.loc['White'])

from scipy.stats import chisquare

chisquare(obs_B,exp_B)

print('Black deaths p-value:{}, \n Hispanic deaths p-value:{}, \n White deaths p-value:{}.'.format(chisquare(obs_B,exp_B)[1],chisquare(obs_H,exp_H)[1],chisquare(obs_W,exp_W)[1]))
from scipy.stats import kruskal

kruskal(n_tab['White'],n_tab['Black'])
data = pd.get_dummies(df[['race','state','region']]) 

ind = ['race_Asian', 'race_Black', 'race_Hispanic', 'race_Native', 'race_Other', 'race_White']

cols_1 = ['region_MW', 'region_NE', 'region_S', 'region_W']

cols_2 = ['state_AK', 'state_AL', 'state_AR','state_AZ', 'state_CA', 'state_CO', 'state_CT', 'state_DC', 'state_DE',

          'state_FL', 'state_GA', 'state_HI', 'state_IA', 'state_ID', 'state_IL', 'state_IN', 'state_KS', 'state_KY', 'state_LA', 

          'state_MA', 'state_MD','state_ME', 'state_MI', 'state_MN', 'state_MO', 'state_MS', 'state_MT', 'state_NC', 'state_ND', 

          'state_NE', 'state_NH', 'state_NJ', 'state_NM','state_NV', 'state_NY', 'state_OH', 'state_OK', 'state_OR', 'state_PA',

          'state_RI', 'state_SC', 'state_SD', 'state_TN', 'state_TX', 'state_UT','state_VA', 'state_VT', 'state_WA', 'state_WI', 

          'state_WV', 'state_WY']
plt.figure(figsize=(10,7))

sns.heatmap(data.corr()[cols_1].loc[ind], annot = True, fmt = ".2f", cmap = "coolwarm")
fig = go.Figure()

fig.add_trace(go.Heatmap(z = data.corr()[cols_2].loc[ind], x=cols_2, y = ind, colorscale = 'thermal'))

fig.update_layout(title_text = 'Race-State deaths correlation table')

fig.show()
US_pop = df_pop['Population'].sum()

Us_mean = len(df)/US_pop

Threshold = 60/Us_mean



low_states = df_pop['State'][df_pop['Population']<Threshold]

other_pop = df_pop['Population'][df_pop['Population']<Threshold].sum()

others_dist = np.zeros(3)

for state in low_states:

    others_dist  = others_dist + np.array(df_pop[['White (mil)', 'Hispanic (mil','Black (mil)']][df_pop['State'] == state])



n_row = {'State':'Other', 'Abbreviation':'OTR', 'Population': other_pop, 'White (mil)': others_dist[0][0],'Hispanic (mil':others_dist[0][1], 'Black (mil)': others_dist[0][1] }

df_pop = df_pop.append(n_row, ignore_index = True)
tab = pd.crosstab(df['race'],df['state'], margins = False)

low_ab = [df_pop['Abbreviation'][df_pop['State']==state] for state in low_states]

new_other = np.zeros(6).astype(int)

for ab in low_ab:

    A = np.array(tab[ab]).T

    new_other = new_other + A

    tab = tab.drop(ab,axis = 1)



n_col = pd.DataFrame({'OTR': new_other[0]})



n_col.index = ['Asian','Black','Hispanic','Native','Other','White']



tab = pd.concat([tab,n_col],axis = 1)



n_tab = pd.concat([tab.iloc[5],tab.iloc[2],tab.iloc[1]],axis = 1)

for state in n_tab.index:

    A = np.array(n_tab.loc[state])

    B = np.array(df_pop[['White (mil)', 'Hispanic (mil','Black (mil)']][df_pop['Abbreviation']== state])

    entries = A/B[0][0]

    n_tab.loc[state] = entries[0],entries[1],entries[2]



n_tab = round(n_tab).astype(int)

n_tab = n_tab[['White','Black']]

n_tab
stat,p,dof,expected = chi2_contingency(n_tab)

p,expected
data = df.groupby('Year')['Month'].count()

fig = go.Figure()

fig.add_trace(go.Bar(x = data.index, y = data))

fig.update_layout(yaxis = dict(title = 'Deaths count'))
fig = go.Figure()

fig.add_trace(go.Box(y = df.groupby('Year')['Month'].value_counts(),boxmean = 'sd', name ='Deaths per month 2015-2020'))

fig.show()

data = df.groupby('Year')['Month'].value_counts()

fig = go.Figure()

fig.add_trace(go.Box(x = [i[0] for i in data.index], y = data,boxmean = True))

fig.update_layout(yaxis = dict(title = 'Deaths count'))

fig.show()
from plotly.subplots import make_subplots



fig = go.Figure()

for year in df['Year'].unique():

    df_year = df[df['Year']==year]

    entry = [(month,df_year['Month'][df_year['Month']==month].count()) for month in df_year['Month'].unique()]

    fig.add_trace(go.Bar(x = [i[0] for i in entry], y = [j[1] for j in entry], name = '{}'.format(year)))

    fig.update_layout(barmode='group', yaxis = dict(title = 'Deaths count'), showlegend = True)

fig.show()
data = df.groupby('Year')['Month'].value_counts().to_frame(name = 'count').reset_index()

nmonth = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}

data['month_n'] = data['Month'].map(nmonth)

data['month_year'] = data['Year'].astype(str)+' '+ data['Month']

fig = go.Figure()

d = data.sort_values(by = ['Year','month_n'])

fig.add_trace(go.Bar(x = d['month_year'], y = d['count'], marker = dict(color =d['count'])))

fig.update_layout(barmode='group', yaxis = dict(title = 'Victims count'), xaxis = dict(nticks = 6, dtick= 12,  tickangle = -40))

fig.show()
data = df.groupby(['Year','race'])['Month'].value_counts().to_frame(name = 'count').reset_index()



data_black = data.loc[(data['race'] == 'Black') & (data['Year'] <= 2017)].copy()

data_white = data.loc[(data['race'] == 'White')& (data['Year'] <= 2017)].copy()

data_hispanic = data.loc[(data['race'] == 'Hispanic') & (data['Year'] <= 2017)].copy()



wbh_dist_2015 = [197.534,39.597,56.496]

wbh_dist_2016 = [197.479, 39.717,57.398]

wbh_dist_2017 = [197.285, 40.129, 58.846]



white_dist = np.append([np.array([19.7534 for i in range(12)]),np.array([19.7479 for i in range(12)])], 

                      [np.array([19.7285 for i in range(12)])])

black_dist = np.append([np.array([3.9597 for i in range(12)]),np.array([3.9717 for i in range(12)])], 

                      [np.array([4.0129 for i in range(12)])])

hisp_dist = np.append([np.array([5.6496 for i in range(12)]),np.array([5.7398 for i in range(12)])], 

                      [np.array([5.8846 for i in range(12)])])



data_white['count'] = data_white['count']/white_dist

data_black['count'] = data_black['count']/black_dist

data_hispanic['count'] = data_hispanic['count']/hisp_dist





print(kruskal(data_white['count'],data_black['count']))

print(kruskal(data_white['count'],data_hispanic['count']))

print(kruskal(data_hispanic['count'],data_black['count']))