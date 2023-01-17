import numpy as np 

import pandas as pd



from matplotlib import pyplot as plt

import seaborn as sns

sns.set()



import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/athlete_events.csv')

noc = pd.read_csv('../input/120-years-of-olympic-history-athletes-and-results/noc_regions.csv')
df.info()
df.head()
noc.info()
noc.head()
df['NOC'].nunique()
df['Team'].nunique()
data = pd.merge(df,noc,how='left',on='NOC')
summer = data.loc[data['Season']=='Summer']

summer.head()
summer.isnull().sum()
summer['Medal'].fillna('DNW', inplace = True)



summer = summer.drop_duplicates()
summer.loc[:, ['NOC', 'Team']].drop_duplicates()['NOC'].value_counts().head(10)
summer.loc[summer['NOC']=='SWE']['Team'].unique()
summer.loc[summer['NOC']=='USA']['Team'].unique()
summer.loc[summer['region'].isnull(),['NOC', 'Team']].drop_duplicates()
summer['region'] = np.where(summer['NOC']=='SGP', 'Singapore', summer['region'])

summer['region'] = np.where(summer['NOC']=='ROT', 'Refugee Olympic Athletes', summer['region'])

summer['region'] = np.where(summer['NOC']=='TUV', 'Tuvalu', summer['region'])

summer['region'] = np.where(summer['NOC']=='UNK', 'Unknown', summer['region'])
summer.drop(['notes','Team'], axis = 1 , inplace = True)

summer.rename(columns = {'region':'Country'}, inplace = True)
summer.loc[:, ['NOC', 'Country']].drop_duplicates()['NOC'].value_counts().head()
summer[['Year','City']].drop_duplicates().sort_values('Year')
country_dict = {'Athina':'Greece',

                'Paris':'France',

                'St. Louis':'USA',

                'London':'UK',

                'Stockholm':"Sweden",

                'Antwerpen':'Belgium',

                'Amsterdam':'Netherlands',

                'Los Angeles':'USA',

               'Berlin':'Germany',

                'Helsinki':'Finland',

                'Melbourne':'Australia',

                'Roma':'Italy',

                'Tokyo':'Japan',

                'Mexico City':'Mexico',

                'Munich':'Germany',

                'Montreal':'Canada',

                'Moskva':'Russia',

                'Seoul':'South Korea',

               'Barcelona':'Spain',

               'Atlanta':'USA',

               'Sydney':'Australia',

               'Beijing':'China',

               'Rio de Janeiro':'Brazil'}
summer['Host_Country']=summer['City'].map(country_dict)

summer.head()
plt.figure(figsize=(12,8))

summer.groupby('Year')['Country'].nunique().plot(kind='bar',color='lightseagreen')

plt.xticks(rotation = 60)

plt.ylabel("Number of Participating Countries")

plt.title("Countries at the Summer Olympic Games")

#plt.savefig('country_by_year')
plt.figure(figsize=(14,14))

plt.subplot(2,1,1)

summer.groupby('Year')['ID'].nunique().plot(color='k',marker='o')

plt.ylabel("Total Number of Athletes")

plt.title("Athletes at the Summer Olympic Games")

plt.subplot(2,1,2)

summer.loc[summer['Sex']=='M'].groupby('Year')['ID'].nunique().plot(color='b',marker='o',label='Male Athletes')

summer.loc[summer['Sex']=='F'].groupby('Year')['ID'].nunique().plot(color='r',marker='o',label='Female Athletes')

plt.ylabel("Number of Athletes")

plt.legend(loc='upper left')

plt.title("Athletes at the Summer Olympic Games by Sex")
most_part = summer.groupby('NOC').nunique()[['Year','NOC']].sort_values('Year',ascending=False).head(20)

most_part.plot(y='Year',kind='barh',legend=None,figsize=(6,8),color=sns.cubehelix_palette(20,start=3,rot=-.25,reverse=True))

plt.gca().invert_yaxis()

plt.xticks(rotation=60)

plt.xlabel('NOC')

plt.ylabel('Number of Editions')

plt.title('Countries with Most Participation')
plt.figure(figsize=(6,6))

hosts = summer[['Year','Host_Country']].drop_duplicates()

sns.countplot(y='Host_Country',data=hosts,order = hosts['Host_Country'].value_counts().index,

              palette=sns.cubehelix_palette(20,start=3,rot=-.25,reverse=True))

plt.xlabel('')
medals = summer.loc[summer['Medal']!='DNW']



medals.head()
medals['Medal_Won'] = 1

team_events = pd.pivot_table(medals,

                            index = ['Country', 'Year', 'Event'],

                                    columns = 'Medal',

                                    values = 'Medal_Won',

                                    aggfunc = 'sum',

                                     fill_value = 0).reset_index()



team_events = team_events.loc[team_events['Gold'] > 1, :]



team_sports = team_events['Event'].unique()



team_sports
team_sports = list(set(team_sports) - set(["Swimming Women's 100 metres Freestyle"," Swimming Men's 50 metres Freestyle",

                                           "Gymnastics Women's Balance Beam","Gymnastics Men's Horizontal Bar"]))
medals['Team_Event'] = np.where(medals.Event.map(lambda x: x in team_sports),1,0)

medals['Individual_Event'] = np.where(medals.Team_Event,0,1)
medals_tally = medals.groupby(['Year', 'NOC', 'Country','Sport','Event', 'Medal'])[['Medal_Won', 'Team_Event','Individual_Event']].agg('sum').reset_index()



medals_tally['Medal_Count'] = medals_tally['Medal_Won']/(medals_tally['Team_Event']+medals_tally['Individual_Event'])
medals_tally['Sex'] = 'M'

medals_tally.loc[medals_tally['Event'].str.contains('Women'),'Sex']='F'



medals_tally.tail()
top_countries = medals_tally.groupby(['Country'])['Medal_Count'].sum().reset_index().sort_values('Medal_Count',ascending=False)



top_countries.head(10).plot(kind='bar',y='Medal_Count',x='Country',legend=None,figsize=(10,6),color=sns.cubehelix_palette(10,reverse=True))

plt.xticks(rotation=60)

plt.xlabel('Country')

plt.ylabel('Medals')

plt.title('Medals by Country')
plt.figure(figsize=(14,14))

plt.subplot(2,1,1)

medals_tally.groupby('Year')['Event'].nunique().plot(color='k',marker='o')

plt.ylabel("Number of Events")

plt.title("Number of Events at the Summer Olympic Games")

plt.subplot(2,1,2)

medals_tally.loc[medals_tally['Sex']=='M'].groupby('Year')['Event'].nunique().plot(marker='o',label='Events for Male Athletes')

medals_tally.loc[medals_tally['Sex']=='F'].groupby('Year')['Event'].nunique().plot(color='r',marker='o',label='Events for Female Athletes')

plt.ylabel("Number of Events")

plt.legend(loc='upper left')

plt.title("Number of Events at the Summer Olympic Games by Sex")
topw = medals_tally.loc[medals_tally['Sex']=='F'].groupby(['Country'])['Medal_Count'].sum().reset_index().sort_values('Medal_Count',ascending=False)

topm = medals_tally.loc[medals_tally['Sex']=='M'].groupby(['Country'])['Medal_Count'].sum().reset_index().sort_values('Medal_Count',ascending=False)

topnew = medals_tally.loc[medals_tally['Year']>1984].groupby(['Country'])['Medal_Count'].sum().reset_index().sort_values(

    'Medal_Count',ascending=False)



topold = medals_tally.loc[medals_tally['Year']<=1984].groupby(['Country'])['Medal_Count'].sum().reset_index().sort_values('Medal_Count',ascending=False)

fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(nrows=2, ncols=2)



topw.head(10).plot(kind='bar',y='Medal_Count',x='Country',legend=None,figsize=(18,12),color=sns.color_palette("YlOrRd_d"),ax=ax1)

ax1.set_xticklabels(labels= topw.head(10)['Country'],rotation=60)

ax1.set_xlabel('Country')

ax1.set_ylabel('Medals Won by Women')

ax1.set_title('Medals by Country (Women)')

topm.head(10).plot(kind='bar',y='Medal_Count',x='Country',legend=None,figsize=(18,12),color=sns.color_palette("Blues_d"),ax=ax2)

ax2.set_xticklabels(labels= topw.head(10)['Country'],rotation=60)

ax2.set_xlabel('Country')

ax2.set_ylabel('Medals Won by Men')

ax2.set_title('Medals by Country (Men)')

topnew.head(10).plot(kind='bar',y='Medal_Count',x='Country',legend=None,figsize=(18,12),color=sns.color_palette("YlOrRd_d"),ax=ax3)

ax3.set_xticklabels(labels= topnew.head(10)['Country'],rotation=60)

ax3.set_xlabel('Country')

ax3.set_ylabel('Medals')

ax3.set_title('Medals by Country (1988 Onwards)')

topold.head(10).plot(kind='bar',y='Medal_Count',x='Country',legend=None,figsize=(18,12),color=sns.color_palette("Blues_d"),ax=ax4)

ax4.set_xticklabels(labels= topold.head(10)['Country'],rotation=60)

ax4.set_xlabel('Country')

ax4.set_ylabel('Medals Won by Men')

ax4.set_title('Medals by Country (Before 1988)')

fig.subplots_adjust(hspace=0.5)

best_countries = ['USA','China','Russia','Germany','Australia','UK']
best_recent = medals_tally.loc[(medals_tally['Country'].map(lambda x: x in best_countries))&(medals_tally['Year']>=1988)].groupby(['Country','NOC','Sport','Sex','Event','Medal'])['Medal_Count'].sum().reset_index()
best_recent.head()
pd.pivot_table(best_recent,

              index = 'Sport',

              columns = 'Country',

              values = 'Medal_Count',

              aggfunc = 'sum',

              fill_value = 0,

              margins = True).sort_values('All', ascending = False)[1:].head(20)

medals_by_type = pd.pivot_table(best_recent,

              index = 'Country',

              columns = 'Medal',

              values = 'Medal_Count',

              aggfunc = 'sum',

              fill_value = 0,

              margins = True).loc[:, ['Gold', 'Silver', 'Bronze']]

medals_by_type.head()
gsb = ['gold','silver','sienna']



medals_by_type[:6].plot(kind = 'bar', stacked = True, figsize = (10,7),color=sns.color_palette(gsb) )

plt.xticks(rotation=0)

plt.ylabel('Medal Count')

plt.xlabel('Country')

plt.title('Medals Breakdown by Country in the Recent Years')
team_size = summer.loc[(summer['Year']>1980)&(summer['Country'].map(lambda x: x in best_countries))].drop_duplicates().groupby(['Year', 'Country']).ID.count().reset_index(name='Team Size')
team_size.head(10)
fig, ((ax1, ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3, ncols=2)



team_size.loc[team_size['Country']=='USA'].plot(y='Team Size',x='Year',

                                                label='Team Size',figsize=(18,20),color='k',marker='o',ax=ax1)

medals_tally.loc[(medals_tally['Year']>1980)&(medals_tally['Country']=='USA')].groupby('Year').sum().plot(y='Medal_Count',

                                                label='Medals',figsize=(18,20),color='b',marker='o',ax=ax1)

ax1.set_xlabel('Year')

ax1.set_ylabel('Team Size/Medals')

ax1.set_title('USA')

team_size.loc[team_size['Country']=='Russia'].plot(y='Team Size',x='Year',

                                                label='Team Size',figsize=(18,20),color='k',marker='o',ax=ax2)

medals_tally.loc[(medals_tally['Year']>1980)&(medals_tally['Country']=='Russia')].groupby('Year').sum().plot(y='Medal_Count',

                                                label='Medals',figsize=(18,20),color='purple',marker='o',ax=ax2)

ax2.set_xlabel('Year')

ax2.set_ylabel('Team Size/Medals')

ax2.set_title('Russia')

team_size.loc[team_size['Country']=='Germany'].plot(y='Team Size',x='Year',

                                                label='Team Size',figsize=(18,20),color='k',marker='o',ax=ax3)

medals_tally.loc[(medals_tally['Year']>1980)&(medals_tally['Country']=='Germany')].groupby('Year').sum().plot(y='Medal_Count',

                                                label='Medals',figsize=(18,20),color='yellow',marker='o',ax=ax3)

ax3.set_xlabel('Year')

ax3.set_ylabel('Team Size/Medals')

ax3.set_title('Germany')

team_size.loc[team_size['Country']=='China'].plot(y='Team Size',x='Year',

                                                label='Team Size',figsize=(18,20),color='k',marker='o',ax=ax4)

medals_tally.loc[(medals_tally['Year']>1980)&(medals_tally['Country']=='China')].groupby('Year').sum().plot(y='Medal_Count',

                                                label='Medals',figsize=(18,20),color='r',marker='o',ax=ax4)

ax4.set_xlabel('Year')

ax4.set_ylabel('Team Size/Medals')

ax4.set_title('China')

team_size.loc[team_size['Country']=='Australia'].plot(y='Team Size',x='Year',

                                                label='Team Size',figsize=(18,20),color='k',marker='o',ax=ax5)

medals_tally.loc[(medals_tally['Year']>1980)&(medals_tally['Country']=='Australia')].groupby('Year').sum().plot(y='Medal_Count',

                                                label='Medals',figsize=(18,20),color='lightseagreen',marker='o',ax=ax5)

ax5.set_xlabel('Year')

ax5.set_ylabel('Team Size/Medals')

ax5.set_title('Australia')

team_size.loc[team_size['Country']=='UK'].plot(y='Team Size',x='Year',

                                                label='Team Size',figsize=(18,20),color='k',marker='o',ax=ax6)

medals_tally.loc[(medals_tally['Year']>1980)&(medals_tally['Country']=='UK')].groupby('Year').sum().plot(y='Medal_Count',

                                                label='Medals',figsize=(18,20),color='royalblue',marker='o',ax=ax6)

ax6.set_xlabel('Year')

ax6.set_ylabel('Team Size/Medals')

ax6.set_title('UK (or GBR)')





fig.subplots_adjust(hspace=0.3)

medals_grouped = pd.DataFrame(medals_tally.groupby(['Year','Country'])['Medal_Count'].sum().reset_index())
host_nations = summer[['Year', 'Host_Country',  'Country']].drop_duplicates()

host_nations = host_nations.loc[host_nations['Host_Country'] == host_nations['Country']]



host_nations['Prev_Year'] = host_nations['Year'] - 4

host_nations['Next_Year'] = host_nations['Year'] + 4



host_medals = host_nations.merge(medals_grouped, left_on=['Year','Country'],right_on=['Year','Country'],how='left')

host_medals.rename(columns = {'Medal_Count':'Medal_Count_Host'},inplace=True)



host_medals = host_medals.merge(medals_grouped, left_on=['Prev_Year','Country'],right_on=['Year','Country'],how='left')

host_medals.drop('Year_y',axis=1,inplace=True)

host_medals.rename(columns = {'Year_x': 'Year','Medal_Count':'Medal_Count_Prev'},inplace=True)



host_medals = host_medals.merge(medals_grouped, left_on=['Next_Year','Country'],right_on=['Year','Country'],how='left')

host_medals.drop('Year_y',axis=1,inplace=True)

host_medals.rename(columns = {'Year_x': 'Year','Medal_Count':'Medal_Count_Next'},inplace=True)



host_medals.head()

host_medals.drop(['Prev_Year','Next_Year'],axis=1,inplace=True)

host_medals = host_medals.sort_values('Year')
host_medals
gdp = pd.read_excel('../input/world-gdp/world_gdp.xls',skiprows=3)

gdp.head()
gdp.columns
gdp = gdp[['Country Name','1988', '1992', '1996', '2000', '2004','2008','2012', '2016']]

gdp.rename(columns={'Country Name':'Country'},inplace=True)

gdp.tail()
gdp = pd.melt(gdp, 

            id_vars='Country', 

            value_vars=list(gdp.columns[1:]), 

            var_name='Year', 

            value_name='GDP')

gdp.sort_values(['Country','Year'],ascending = [True,True],inplace=True)

gdp.head()
gdp['Year'] = gdp['Year'].astype(int)



set(medals_tally['Country']) - set(gdp['Country'])
set(gdp['Country']) - set(medals_tally['Country'])
to_replace = ['Bahamas, The','Egypt, Arab Rep.','Iran, Islamic Rep.',"Cote d'Ivoire",'Kyrgyz Republic','North Macedonia',

             'Korea, Dem. People’s Rep.','Russian Federation','Slovak Republic','Korea, Rep.','Syrian Arab Republic',

              'Trinidad and Tobago','United Kingdom','United States','Venezuela, RB','Virgin Islands (U.S.)']

            



new_countries =   ['Bahamas','Egypt', 'Iran', 'Ivory Coast','Kyrgyzstan','Macedonia','North Korea','Russia','Slovakia',

                   'South Korea','Syria','Trinidad','UK','USA','Venezuela','Virgin Islands, US']



gdp.replace(to_replace,new_countries,inplace=True)
medals_by_country = medals_tally.loc[medals_tally['Year']>1984].groupby(['Year','NOC','Country'])['Medal_Count'].sum().reset_index()

medals_by_country.head()
medals_tally_gdp = medals_by_country.merge(gdp,

                                   left_on = ['Year', 'Country'],

                                   right_on = ['Year', 'Country'],

                                   how = 'left')
event_yrs = [1988,1992,1996,2000,2004,2008,2012,2016]

tw_gdp = [165.6,254.2,358.7,481.0,605.1,804.8,984.4,1132.9]

tw_dict =dict(zip(event_yrs,tw_gdp))



medals_tally_gdp.loc[medals_tally_gdp.Country=='Taiwan','GDP'] = medals_tally_gdp.loc[medals_tally_gdp.Country=='Taiwan','Year'].map(tw_dict)
medals_tally_gdp.isnull().sum()/medals_tally_gdp.count()
team_size = summer.loc[summer.Year >= 1988].drop_duplicates().groupby(['Year', 'Country']).ID.count().reset_index(name='Team_Size')
train = medals_tally_gdp.merge(team_size,left_on=['Year','Country'],right_on=['Year','Country'],how='left')

train.head()
pop = pd.read_csv('../input/countries-population/WorldPopulation.csv',

                  usecols=['Country','1988','1992','1996','2000','2004','2008','2012','2016'])



pop.head()
pop = pd.melt(pop,

              id_vars='Country', 

            value_vars=list(pop.columns[1:]), 

            var_name='Year', 

            value_name='Population')



pop['Year'] = pop['Year'].astype(int)
train= train.merge(pop,on=['Year','Country'],how='left')



train.head()
sns.pairplot(train,vars=['Medal_Count','GDP','Team_Size','Population'])
train['Log_GDP'] = train['GDP'].apply(np.log)

train['Log_Population'] = train['Population'].apply(np.log)

train['Log_GDP_PC'] = train['Log_GDP'] - train['Log_Population'] #because log(gdp/pop) = log(gdp) - log(pop)
corr = train.loc[train['Medal_Count']>10.0,['Log_GDP', 'Medal_Count']].corr()['Medal_Count'][0]



plt.plot(train.loc[train['Medal_Count']>10.0,  'Log_GDP'], 

     train.loc[train['Medal_Count']>10.0,  'Medal_Count'] , 

     linestyle = 'none', 

     marker = 'o',

    color = 'navy',

    alpha = 0.6)

plt.xlabel('Log(GDP)')

plt.ylabel('Number of Medals')

plt.title('Log(GDP) versus medal tally')

plt.text(22.7, 

     115,

     "Correlation = " + str(corr))
corr = train.loc[train['Medal_Count']>10.0,['Log_Population', 'Medal_Count']].corr()['Medal_Count'][0]



plt.plot(train.loc[train['Medal_Count']>10.0,  'Log_Population'], 

     train.loc[train['Medal_Count']>10.0,  'Medal_Count'] , 

     linestyle = 'none', 

     marker = 'o',

    color = 'lightseagreen',

    alpha = 0.9)

plt.xlabel('log(Population)')

plt.ylabel('Number of Medals')

plt.title('log(Population) versus medal tally')

plt.text(14.8, 

     120,

     "Correlation = " + str(corr))
corr = train.loc[train['Medal_Count']>10.0,['Log_GDP_PC', 'Medal_Count']].corr()['Medal_Count'][0]



plt.plot(train.loc[train['Medal_Count']>10.0,  'Log_GDP_PC'], 

     train.loc[train['Medal_Count']>10.0,  'Medal_Count'] , 

     linestyle = 'none', 

     marker = 'o',

    alpha = 0.9)

plt.xlabel('Logarithm of GDP Per Capita')

plt.ylabel('Number of Medals')

plt.title('log(GDP Per Capita) versus medal tally')

plt.text(5.8, 

     117,

     "Correlation = " + str(corr))
corr = train.loc[train['Medal_Count']>10.0,['Team_Size', 'Medal_Count']].corr()['Medal_Count'][0]



plt.plot(train.loc[train['Medal_Count']>10.0,  'Team_Size'], 

     train.loc[train['Medal_Count']>10.0,  'Medal_Count'] , 

     linestyle = 'none', 

     marker = 'o',

         color='r',

    alpha = 0.8)

plt.xlabel('Team Size')

plt.ylabel('Number of Medals')

plt.title('Team Size versus medal tally')

plt.text(50,120,

     "Correlation = " + str(corr))
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error as mse

import statsmodels.formula.api as smf
#train models on data upto 2012

X_tr = train.loc[train.Year != 2016].dropna()[['Team_Size','Log_GDP','Log_Population']]

y_tr = train.loc[train.Year != 2016].dropna()['Medal_Count']

#predict on 2016

X_tst = train.loc[train.Year == 2016].dropna()[['Team_Size','Log_GDP','Log_Population']]

y_tst = train.loc[train.Year == 2016].dropna()['Medal_Count']
lr = LinearRegression()

lr.fit(X_tr,y_tr)

y_pred = lr.predict(X_tst)



lr_score = lr.score(X_tst,y_tst) #this gives the R^2 score

lr_err = np.sqrt(mse(y_tst,y_pred)) #this gives the rms error



print('Linear Regression R^2: {}, Linear Regression RMSE: {}'.format(lr_score,lr_err))
OLS = smf.ols('Medal_Count ~ Team_Size + Log_GDP + Log_Population', data=train.loc[train.Year!=2016]).fit()



y_ols = OLS.predict(X_tst)

ols_score = OLS.rsquared #R^2

ols_err = np.sqrt(mse(y_tst, y_ols)) #rms error

print('Statsmodels OLS R^2: {}, Statsmodels OLS RMSE: {}'.format(ols_score,ols_err))