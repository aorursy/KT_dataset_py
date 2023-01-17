# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import plotly.offline as py
py.init_notebook_mode(connected=True)
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
color = sns.color_palette()
import matplotlib.pylab as plt
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
import plotly.tools as tls



def check_missing(df):
    
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)
    missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
    

def find_uni(df):
    col_list = df.columns
    redundant_col =[]
    for col in col_list:
        if df[col].nunique() == 1:
            redundant_col.append(col)
    return redundant_col

# Any results you write to the current directory are saved as output.

DIR = sorted(os.listdir("../input")) ;
print(DIR)
olp = pd.read_csv('../input/' + DIR[0] + '/athlete_events.csv')
# Lets read in the noc_country mapping first
noc_country = pd.read_csv('../input/' + DIR[0] + '/noc_regions.csv')
gdp = pd.read_csv('../input/' + DIR[1] + '/world_gdp.csv', skiprows = 3)
pop = pd.read_csv('../input/' + DIR[2] + '/world_pop.csv')
# print(df)
print ('Size of the World GDP data : ',gdp.shape)
print ('Size of the World Poplation data : ',pop.shape)
print ('Size of the health-inspections data : ',olp.shape)


noc_country.drop('notes', axis = 1 , inplace = True)
noc_country.rename(columns = {'region':'Country'}, inplace = True)

noc_country.head()


# Remove unnecessary columns
missing_data_df = check_missing(gdp)
missing_data_df.head()
redundant_col = find_uni(gdp)
gdp.drop(redundant_col,axis=1,inplace =True)
# gdp.drop(['Indicator Name', 'Indicator Code'], axis = 1, inplace = True)
# The columns are the years for which the GDP has been recorded. This needs to brought into a single column for efficient
# merging.
gdp = pd.melt(gdp, id_vars = ['Country Name', 'Country Code'], var_name = 'Year', value_name = 'GDP')
# convert the year column to numeric
gdp['Year'] = pd.to_numeric(gdp['Year'])


# Check the duplicate feature.
missing_data_df = check_missing(pop)
missing_data_df.head()
redundant_col = find_uni(pop)
# So, these two features are redundant for the ML algorithm.
print ('Number of redundant features in data :',len(redundant_col))
print ('Redundant Feature :', redundant_col)
print(pop[redundant_col[0]].value_counts(),pop[redundant_col[1]].value_counts()
)
pop.drop(redundant_col,axis=1,inplace =True)
pop = pd.melt(pop, id_vars = ['Country', 'Country Code'], var_name = 'Year', value_name = 'Population')
# Change the Year to integer type
pop['Year'] = pd.to_numeric(pop['Year'])

olp['Medal'].fillna('DNW', inplace = True)
# olp.drop('NOC' , axis =1, inplace =True)

# Merge to get country code
olp_merge = olp.merge(noc_country,
                                left_on = 'NOC',
                                right_on = 'NOC',
                                how = 'left')
# Merge to get country code
olp_merge_ccode = olp_merge.merge(gdp[['Country Name', 'Country Code']].drop_duplicates(),
                                            left_on = 'Team',
                                            right_on = 'Country Name',
                                            how = 'left')
olp_merge_ccode.drop('Country Name', axis = 1, inplace = True)

# Merge to get gdp too
olp_merge_gdp = olp_merge_ccode.merge(gdp,
                                        left_on = ['Country Code', 'Year'],
                                        right_on = ['Country Code', 'Year'],
                                        how = 'left')

olp_merge_gdp.drop('Country Name', axis = 1, inplace = True)
cpl = olp_merge_gdp.merge(pop,
                            left_on = ['Country Code', 'Year'],
                            right_on= ['Country Code', 'Year'],
                            how = 'left')
missing_data_df = check_missing(cpl)
missing_data_df.head()
redundant_col = find_uni(cpl)
cpl.drop(redundant_col,axis=1,inplace =True)
cpl.drop(['Country_x','Country_y'], axis = 1, inplace = True)

cpl.head()



# print(df.head(0))
cat_olp = olp.select_dtypes(include = 'object').columns.tolist()
num_olp = olp.select_dtypes(exclude='object').columns.tolist()
print ('\n 120 Years Olympic categorical feature :', cat_olp)
print ('\n 120 Years Olympic numeric feature :' ,num_olp)
print ('\n 120 Years Olympic number of categorical feature : ' , len(cat_olp))
print ('\n 120 Years Olympic number of numeric feature : ' , len(num_olp))


cat_pop = pop.select_dtypes(include = 'object').columns.tolist()
num_pop = pop.select_dtypes(exclude='object').columns.tolist()
print ('\n World Population categorical feature :', cat_pop)
print ('\n World Population numeric feature :' ,num_pop)
print ('\n World Population number of categorical feature : ' , len(cat_pop))
print ('\n World Population number of numeric feature : ' , len(num_pop))


cat_gdp = gdp.select_dtypes(include = 'object').columns.tolist()
num_gdp = gdp.select_dtypes(exclude='object').columns.tolist()
print ('\n World GDP categorical feature :', cat_gdp)
print ('\n World GDP numeric feature :' ,num_gdp)
print ('\n World GDP number of categorical feature : ' , len(cat_gdp))
print ('\n World GDP number of numeric feature : ' , len(num_gdp))


len(list(set(olp_merge['NOC'].unique()) - set(gdp['Country Code'].unique())))
len(list(set(olp_merge['Team'].unique()) - set(gdp['Country Name'].unique())))

olp = olp[np.isfinite(olp['Age'])]
plt.figure(figsize=(20, 10))
plt.tight_layout()
plt.subplot(4,1,1)
sns.countplot(olp['Age'])

plt.subplot(4,1,2)
gold = olp[olp['Medal'] == 'Gold'] 
sns.countplot(gold['Age'])

plt.subplot(4,1,3)
silver = olp[olp['Medal'] == 'Silver'] 
sns.countplot(silver['Age'])

plt.subplot(4,1,4)
bronze = olp[olp['Medal'] == 'Bronze'] 
sns.countplot(bronze['Age'])

# It is best chance to have medal in your early 20s
G50 = gold['Sport'][gold['Age'] > 50] ;
G20 = gold['Sport'][gold['Age'] < 20].value_counts().head(10) ;
plt.figure(figsize=(20, 10))
plt.tight_layout()
sns.countplot(G50)
plt.title('>50 year-old Sports & Age')

trace = go.Bar(
    x = G20.index,
    y = G20.values,
)
data = [trace]
layout = go.Layout(
    title = "<20 year-old Sports & Age",
    xaxis=dict(
        title='Sports',
        tickfont=dict(
            size=10,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='People counts',
        titlefont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
resp = py.iplot(fig, filename='G20')

gold.info()

notNullgolds = gold[(gold['Height'].notnull()) & (gold['Weight'].notnull())]
notNullgolds.head()
plt.figure(figsize=(12, 10))
# ax = sns.scatterplot(x="Height", y="Weight", data=notNullgolds)
# ax.grid()

# ax = sns.pairplot(notNullgolds)
from scipy import stats
ax = sns.jointplot(x="Height", y="Weight", 
                   data=notNullgolds,kind="reg", 
                    stat_func=stats.pearsonr, 
                      color=None, height=10, ratio=5, 
                      space=.2, dropna=True, xlim=None, 
                      ylim=None, joint_kws=None, 
                      marginal_kws=None, annot_kws=None)
g = ax.plot_joint(sns.kdeplot)

plt.title('Golden Athletes\' Weight & Height')
gdp.info()
# it is inspired by sagar chadha
# Lets take data from 1961 onwards only and for summer olympics only
cpl_2 = cpl.loc[(cpl['Year'] > 1960) & (cpl['Season'] == "Summer"), :]
# Reset row indices
cpl_2 = cpl_2.reset_index()
cpl_2['Medal_Won'] = np.where(cpl_2.loc[:,'Medal'] == 'DNW', 0, 1)
cpl_2.info()
medals = pd.pivot_table(cpl_2,
                        index = ['Team', 'Year', 'Event'],
                        columns = 'Medal',
                        values = 'Medal_Won',
                        aggfunc = 'sum',
                        fill_value = 0).drop('DNW' , axis =1).reset_index()
medals = medals.loc[medals['Gold'] > 1, :]

team_sports = medals['Event']
medals.head(20)



remove_sports = ["Gymnastics Women's Balance Beam", "Gymnastics Men's Horizontal Bar", 
                 "Swimming Women's 100 metres Freestyle", "Swimming Men's 50 metres Freestyle"]

team_sports = list(set(team_sports) - set(remove_sports))# if an event name matches with one in team sports, then it is a team event. Others are singles events.

team_event_mask = cpl_2['Event'].map(lambda x: x in team_sports)
single_event_mask = [not i for i in team_event_mask]

# rows where medal_won is 1
medal_mask = cpl_2['Medal_Won'] == 1

# Put 1 under team event if medal is won and event in team event list
cpl_2['Team_Event'] = np.where(team_event_mask & medal_mask, 1, 0)

# Put 1 under singles event if medal is won and event not in team event list
cpl_2['Single_Event'] = np.where(single_event_mask & medal_mask, 1, 0)

# Add an identifier for team/single event
cpl_2['Event_Category'] = cpl_2['Single_Event'] + cpl_2['Team_Event']
medal_tally_agnostic = cpl_2.groupby(['Year', 'Team', 'Event', 'Medal'])[['Medal_Won', 'Event_Category']].\
agg('sum').reset_index()

medal_tally_agnostic['Medal_Won_Corrected'] = medal_tally_agnostic['Medal_Won']/medal_tally_agnostic['Event_Category']

# Medal Tally.
medal_tally = medal_tally_agnostic.groupby(['Year','Team'])['Medal_Won_Corrected'].agg('sum').reset_index()

medal_tally_pivot = pd.pivot_table(medal_tally,
                     index = 'Team',
                     columns = 'Year',
                     values = 'Medal_Won_Corrected',
                     aggfunc = 'sum',
                     margins = True).sort_values('All', ascending = False)[1:5]

# print total medals won in the given period
medal_tally_pivot.loc[:,'All']

medal_tally = medal_tally_agnostic.groupby(['Year','Team'])['Medal_Won_Corrected'].agg('sum').reset_index()

year_team_gdp = cpl_2.loc[:, ['Year', 'Team', 'GDP']].drop_duplicates()

medal_tally_gdp = medal_tally.merge(year_team_gdp,
                                   left_on = ['Year', 'Team'],
                                   right_on = ['Year', 'Team'],
                                   how = 'left')

row_mask_5 = medal_tally_gdp['Medal_Won_Corrected'] > 0
top_countries = ['USA', 'Russia', 'Germany', 'China']
row_mask_6 = medal_tally_gdp['Team'].map(lambda x: x in top_countries)

correlation = medal_tally_gdp.loc[row_mask_5, ['GDP', 'Medal_Won_Corrected']].corr()['Medal_Won_Corrected'][0]

plt.plot(medal_tally_gdp.loc[row_mask_5, 'GDP'], 
     medal_tally_gdp.loc[row_mask_5, 'Medal_Won_Corrected'] , 
     linestyle = 'none', 
     marker = 'o',
    alpha = 0.4)
plt.xlabel('Country GDP')

plt.ylabel('Number of Medals')
plt.title('GDP versus medal tally')
plt.text(np.nanpercentile(medal_tally_gdp['GDP'], 99.6), 
     max(medal_tally_gdp['Medal_Won_Corrected']) - 50,
     "Correlation = " + str(correlation))