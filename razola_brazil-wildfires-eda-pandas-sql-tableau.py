# Import libraries
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
%matplotlib inline
import pandasql as psql
sql=psql.sqldf
import numpy as np
sns.set_style("dark")
!pip install sidetable
import sidetable


# Data Import
df = pd.read_csv('../input/forest-fires-in-brazil/amazon.csv', encoding='latin1')
df.head(5)
#Just renaming the number columns
df.rename(columns={'number':'number_fires'},inplace=True)

#Getting the columns into a list
colunas=df.columns.to_list()

#It was messing up my charts :D so I converted to string - but you should use pd.datetime for sure!
df['year']=df['year'].astype(str)

def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum()/data.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return(np.transpose(tt))

missing=missing_data(df)


print(colunas)
print(missing)
#Mapping states by region 
df['state'].unique()

state_region={'Acre':'North', 'Alagoas':'Northeast', 'Amapa':'North', 'Amazonas':'North', 'Bahia':'Northeast', 'Ceara':'Northeast',
       'Distrito Federal':'Midwest', 'Espirito Santo':'Southeast', 'Goias':'Midwest', 'Maranhao':'Northeast',
       'Mato Grosso':'Midwest', 'Minas Gerais':'Southeast', 'Pará':'North', 'Paraiba':'Northeast', 'Pernambuco':'Northeast',
       'Piau':'Northeast', 'Rio':'Southeast', 'Rondonia':'North', 'Roraima':'North', 'Santa Catarina':'South',
       'Sao Paulo':'Southeast', 'Sergipe':'Northeast', 'Tocantins':'North'}
df['region']=df['state'].map(state_region)

#Ordering Months
month_order={'Janeiro':'01', 'Fevereiro':'02', 'Março':'03', 'Abril':'04', 'Maio':'05', 'Junho':'06', 'Julho':'07',
       'Agosto':'08', 'Setembro':'09', 'Outubro':'10', 'Novembro':'11', 'Dezembro':'12'}
df['month_order']=df['month'].map(month_order)
#Using Sidetable to check how data is distributed.

df.stb.freq(['region'],style=True)
#Now that we have this information lets see how many fires by region and state.

fires_by_region=sql('''select region, round(sum(number_fires)) as number_fires, round(avg(number_fires)) as average_fires from df
             group by region
             order by number_fires desc''')   



print(fires_by_region)

#Lets see the distribution by state & Region Now - And Take Only TOP 10

#Using a little bit of SQL because Why Not?
fires_by_regionandstate=sql('''select region,state,round(sum(number_fires)) as number_fires from df
                          group by region,state
                          order by number_fires desc
                          limit 10''')
print(fires_by_regionandstate)
#Let's check the fires trend by region & year


#Coming back to Pandas because Why Not?

evolutionbyregion=df.groupby(['year','region']).sum().sort_values('year',ascending=True).unstack().fillna(0)         

evolutionbyregion.plot(kind='bar',stacked=True,figsize=(25,9),colormap='Accent')
plt.title('Evolution of Fires by Region', fontsize = 25)
plt.xlabel('Year', fontsize = 20)
plt.ylabel('Number of Fires', fontsize = 20)
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(['Midwest','North','Northeast','South','Southeast'],bbox_to_anchor=(1, 0.5))

#Let's see how it is the trend per month


#Sum of all the years



forchartline=df.groupby(['year','region']).sum().sort_values('year',ascending=True).reset_index()
c=sns.relplot(x="year", y="number_fires",
                 col="region",
                 kind="line", data=forchartline)
c.set_xticklabels(rotation=90)

forchartlinemonth=sql('''select region, month_order, sum(number_fires) as number_fires from df
                      group by region,month_order
                      order by month''')
                      
                                    
g=sns.relplot(x='month_order',y='number_fires',col='region',kind='line',data=forchartlinemonth)
g.set_xticklabels(rotation=90)
#We need to identify which party was on power during the period we have available.

party_of_president={'1998':'PSDB', '1999':'PSDB', '2000':'PSDB', '2001':'PSDB', '2002':'PSDB', '2003':'PT', '2004':'PT', '2005':'PT',
       '2006':'PT', '2007':'PT', '2008':'PT', '2009':'PT', '2010':'PT', '2011':'PT', '2012':'PT', '2013':'PT',
       '2014':'PT', '2015':'PT', '2016':'PT', '2017':'PMDB'}

president={'1998':'FHC', '1999':'FHC', '2000':'FHC', '2001':'FHC', '2002':'Lula', '2003':'Lula', '2004':'Lula', '2005':'Lula',
       '2006':'Lula', '2007':'Lula', '2008':'Lula', '2009':'Lula', '2010':'Lula', '2011':'Dilma', '2012':'Dilma', '2013':'Dilma',
       '2014':'Dilma', '2015':'Dilma', '2016':'Dilma', '2017':'Temer'}

df['party']=df['year'].map(party_of_president)
df['president']=df['year'].map(president)

print(df.head(1))

party_year=df.groupby(['year','party','president']).mean().reset_index()

chart = sns.catplot(x="year", y="number_fires",
                hue="president",kind='bar',
                data=party_year,aspect=3)
plt.title('Average Number of Fires by Year and President', fontsize = 25)

#filtering our data to have only Mato Grosso
mt=df[df['state']=='Mato Grosso']

partidos={'1998':'PSDB', '1999':'PSDB', '2000':'PSDB', '2001':'PSDB', '2002':'PSDB', '2003':'Cidadania/Maggi', '2004':'Cidadania/Maggi', '2005':'Cidadania/Maggi',
       '2006':'Cidadania/Maggi', '2007':'Cidadania/Maggi', '2008':'Cidadania/Maggi', '2009':'Cidadania/Maggi', '2010':'PMDB', '2011':'PMDB', '2012':'PMDB', '2013':'PMDB',
       '2014':'PMDB', '2015':'PSDB', '2016':'PSDB', '2017':'PSDB'}

mt['party_state']=mt['year'].map(partidos)

party_year_state=mt.groupby(['year','party_state']).mean().reset_index()

chart2 = sns.catplot(x="year", y="number_fires",
                hue="party_state",kind='bar',
                data=party_year_state,aspect=3)
plt.title('Average Number of Fires by Year and Party in Mato Grosso', fontsize = 25)