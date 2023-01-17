import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display_html 
from scipy import stats
import re
import os
df_electricity = dict()
for file in os.listdir("../input/dutch-energy/Electricity"):
    company = file.split('_')[0]
    year = re.findall('2+[0-9]+',file.split('_')[2])[0]
    df_electricity[company+year] = pd.read_csv("../input/dutch-energy/Electricity/"+file)
df_electricity.keys(),len(df_electricity)
df_electricity['enexis2010'].info()
largest_municipalities = ['AMSTERDAM','ROTTERDAM',"'S-GRAVENHAGE",'UTRECHT',\
                          'EINDHOVEN','TILBURG','ALMERE','GRONINGEN','BREDA','NIJMEGEN']
col1,col2,col3,col4 = [],[],[],[]
for net_manager in df_electricity.keys():
    for city in df_electricity[net_manager].groupby('city').sum().index.values:
        if city in largest_municipalities:
            value = df_electricity[net_manager].groupby('city').sum().annual_consume[city]   
            col1.append(re.findall('^[a-z]+',net_manager)[0])
            col2.append(re.findall('[0-9]+',net_manager)[0])
            col3.append(city)
            col4.append(value)
            
d={'net_manager': col1,
    'year': col2,
    'city': col3,
    'annual_consume': col4}

table = pd.DataFrame.from_dict(data=d)
table_yearcity = table.groupby(['year','city']).sum()['annual_consume'].unstack(level=1)
table_yearcity
fig = plt.figure(figsize=(12,6))
for city in largest_municipalities:
    try:
        plt.plot(table_yearcity[city].index.values,\
                 table_yearcity[city].values, label=city, marker='o')
        plt.legend(loc='center left',bbox_to_anchor=(1, 0.5),fontsize=16)
    except:
        pass
plt.title("Annual consumption of Electricity",{'size':'18'})
plt.ylabel('Annual Consume (kWh)',{'size':'16'})
plt.xlabel('Year',{'size':'16'})
plt.grid(True,color='whitesmoke')
plt.tick_params(axis='both', labelsize=15)

caption = '''The consumption of electricity in kWh per year from 2009 to 2018. Each city in the 10 largest 
dutch municipalities list is represented by a different color. The dots indicate the exact annual consume 
for each city per year, and they are connected by straight lines to better infere the data behavior along 
the years. Notice that in 2009 some of the cities have missing data, therefore no record is shown.'''
fig.text(.5,- .17, caption, ha='center',fontsize=16)

plt.show()
fig = plt.figure(figsize=(12,6))
for net_manager in sorted(df_electricity.keys()):
    if (df_electricity[net_manager].city=='EINDHOVEN').sum() !=0:
        n_conn_street = df_electricity[net_manager][df_electricity[net_manager].city=='EINDHOVEN']\
        .groupby('street').num_connections.sum()
        n_streets = len(n_conn_street)
        n_conn = df_electricity[net_manager][df_electricity[net_manager].city=='EINDHOVEN']\
        .num_connections.sum()
        year = re.findall('[0-9]+',net_manager)[0]        
        plt.barh(year,width=n_conn,label=n_streets)
        plt.legend(title="Number of Streets",loc=0,fontsize=13)

        #print(year,n_conn,n_streets)
        
plt.xlabel('Number of Connections',{'size':'16'})
plt.ylabel('Year',{'size':'16'})
plt.title('Number of Connections per Year in Eindhoven',{'size':'18'})
plt.grid(True,color='whitesmoke')

plt.tick_params(axis='both', labelsize=13)
caption = '''The number of connections per year in Eindhoven. These connections are grouped by ranges of zipcodes,
thus collecting data from different streets. Here, the variable year is treated as a category, which is represented
by different colors. For each year, we also provide the number of distinct streets within the ranges of zipcodes,
as shown by the box inside the plot. For example, the number of connections in 2010 (blue) are from records 
of three distinct streets.'''
fig.text(.5,- .17, caption, ha='center',fontsize=14)

plt.show()
col1,col2,col3 = [],[],[]
        
for i in df_electricity.keys():
    counts = df_electricity[i].groupby('city').num_connections.sum().values
    cities = df_electricity[i].groupby('city').num_connections.sum().index
    year = re.findall('[0-9]+',i)[0]
    for city,count in zip(cities,counts):
        if city in largest_municipalities:
            col1.append(year)
            col2.append(city)
            col3.append(count)
            
d={ 'year': col1,
    'city': col2,
    'n_connections': col3}

table_n_conn = pd.DataFrame.from_dict(data=d)
table_n_connections = table_n_conn.groupby(['year','city']).sum()['n_connections'].unstack(0)
table_n_connections
year_conn_cons =pd.DataFrame({'year':table_n_connections.sum().index,\
              'n_connections':table_n_connections.sum().values,\
              'annual_consume':table_yearcity.T.sum().values})
year_conn_cons
fig = plt.figure(figsize=(12,6))
plt.grid(color='whitesmoke')

xi=year_conn_cons.n_connections
y=year_conn_cons.annual_consume

slope, intercept, r_value, p_value, std_err = stats.linregress(xi,y)
line = slope*xi+intercept
print('slope:',slope,'intercept:',intercept)

for i,j in zip(year_conn_cons.n_connections,year_conn_cons.annual_consume):
    plt.scatter(i,j,label='year')
    plt.legend(title='Year',labels=list(range(2009,2019,1)),loc=4,fontsize=14)

plt.plot(xi, line,color='black',linestyle='--')

plt.xlabel('Number of Connections',{'size':'16'})
plt.ylabel('Annual Consume',{'size':'16'})
plt.title('Total Number of Connections vs Total Annual Consume Per Year',{'size':'18'})
plt.tick_params(axis='both', labelsize=13)

caption = '''Total annual consume of electricity per year vs the total number of connections per year. 
Each different color (dots) labels a different year. The dashed line represents a linear regression for 
these data points, where the slope and the intercept are ~95.61 and ~77821380.46, respectively.'''
fig.text(.5,- .10, caption, ha='center',fontsize=14)

plt.show()
data_not100,data_100,list_year = [],[],[]

for net_manager in sorted(df_electricity.keys()):
    for city in largest_municipalities:
        try:
            year = re.findall('[0-9]+',net_manager)[0]
            data = df_electricity[net_manager].groupby(['city','delivery_perc'])\
            .count()['net_manager'].unstack(level=0)[city]
            list_year.append(year)
            data_not100.append(data[data.index!=100].sum())
            data_100.append(data[data.index==100].values[0])
        except:
            pass
data_plot = pd.DataFrame({"year":list_year,"100":data_100,"Not 100":data_not100})
data_plot[data_plot.year=='2010']
sns.set_style("whitegrid",{'grid.color': '.9'})
sns.catplot(x="100", y="year", kind="box", orient="h", height=6, aspect=1.5,data=data_plot)
plt.xlabel('100',{'size':'16'})
plt.ylabel('Year',{'size':'16'})
plt.title('Counting the electricity delivery % along the years',{'size':'18'});
sns.set_style("whitegrid",{'grid.color': '.9'})
sns.catplot(x="Not 100", y="year", kind="box", orient="h", height=6,aspect=1.5, data=data_plot)
plt.xlabel('Not 100',{'size':'16'})
plt.ylabel('Year',{'size':'16'})
plt.title('Counting the electricity delivery % along the years',{'size':'18'});
d2018 = data_plot[data_plot.year=='2018']
sns.set_style("whitegrid",{'grid.color': '.9'})
sns.catplot(data=d2018[['100','Not 100']],kind="box", orient="h", height=3.2, aspect=3)
plt.xlabel('Count',{'size':'16'})
plt.title('Comparing the electricity delivery % in 2018',{'size':'16'});
dict_enexis, dict_liander, dict_stedin = dict(),dict(),dict() 
dict_enexis_cities, dict_liander_cities, dict_stedin_cities = dict(),dict(),dict() 

for net_manager in sorted(df_electricity.keys()):
    city_names = list(df_electricity[net_manager].city.value_counts().index)
    city_values = list(df_electricity[net_manager].city.value_counts().values)
    year = re.findall('[0-9]+',net_manager)[0]
    net_manager_name = re.findall('^[a-z]+',net_manager)[0]
    #print(year,net_manager_name,len(city_names))
    if net_manager_name == 'enexis':
        dict_enexis[year]=len(city_names)
        dict_enexis_cities[year]=city_names
    elif net_manager_name == 'liander':
        dict_liander[year]=len(city_names)
        dict_liander_cities[year]=city_names
    else:
        dict_stedin[year]=len(city_names)
        dict_stedin_cities[year]=city_names
df_nm = pd.DataFrame.from_dict({'enexis': dict_enexis,'enexis_cities': dict_enexis_cities,\
                        'liander': dict_liander,'liander_cities': dict_liander_cities,\
                        'stedin': dict_stedin,'stedin_cities': dict_stedin_cities})
data_net_managers =\
pd.DataFrame({'enexis': dict_enexis, 'liander': dict_liander,'stedin': dict_stedin})

rescaled_dnm = \
(data_net_managers-data_net_managers.min())/(data_net_managers.max()-data_net_managers.min())

tb1 = data_net_managers.style.set_table_attributes("style='display:inline'")\
.set_caption('Original Data')
tb2= rescaled_dnm.style.set_table_attributes("style='display:inline'")\
.set_caption('Rescaled Data')

display_html(tb1._repr_html_() + tb2._repr_html_(), raw=True)
fig = plt.figure(figsize=(12,6))

plt.scatter(y=rescaled_dnm.enexis,x=data_net_managers.index, marker='d',label='Enexis')
plt.scatter(y=rescaled_dnm.liander,x=data_net_managers.index,label='Liander', marker='8')
plt.scatter(y=rescaled_dnm.stedin,x=data_net_managers.index,label='Stedin', marker='x')

plt.ylabel('Relative # of cities covered',{'size':'16'})
plt.xlabel('Year',{'size':'16'})
plt.title('Relative number of Cities covered per Year by Net Manager',{'size':'18'})
plt.legend(loc=0,fontsize=13)

plt.tick_params(axis='both', labelsize=13)
caption = '''The rescaled data concerning the number of cities covered by each net manager
(Enexis, Liander and Stedin) per year. Here, 0.0 and 1.0 represent the minimum and maximum 
number of cities, respectively. There is no data from Enexis in 2009. '''
fig.text(.5,- .10, caption, ha='center',fontsize=14)

plt.show()
np.array(df_nm[df_nm.index=='2012'].enexis_cities[0].sort(reverse=False)) ==\
np.array(df_nm[df_nm.index=='2013'].enexis_cities[0].sort(reverse=False))
for city in df_nm[df_nm.index=='2012'].liander_cities[0]:
    if city not in df_nm[df_nm.index=='2013'].liander_cities[0]:
        print('2012',city)
for city in df_nm[df_nm.index=='2013'].liander_cities[0]:
    if city not in df_nm[df_nm.index=='2012'].liander_cities[0]:
        print('2013',city)
for city in df_nm[df_nm.index=='2012'].stedin_cities[0]:
    if city not in df_nm[df_nm.index=='2013'].stedin_cities[0]:
        print('2012',city)        
for city in df_nm[df_nm.index=='2013'].stedin_cities[0]:
    if city not in df_nm[df_nm.index=='2012'].stedin_cities[0]:
        print('2013',city)
fig, axs = plt.subplots(1, 3, figsize= (15,3))

for i,j in zip(range(3),['liander','stedin','enexis']):
    axs[i].hist(df_electricity[j+'2018'].annual_consume_lowtarif_perc,alpha=0.5,label='2018')
    axs[i].axvline(x=df_electricity[j+'2018'].annual_consume_lowtarif_perc.describe()['mean'])
    axs[i].hist(df_electricity[j+'2010'].annual_consume_lowtarif_perc,alpha=0.5,label='2010')
    axs[i].axvline(x=df_electricity[j+'2010'].annual_consume_lowtarif_perc.describe()['mean'],\
               color='orange',linestyle='--')
    axs[i].set_title(j.capitalize(),{'size':'18'})

axs[1].set_xlabel('% of consume during the low tarif hours',{'size':'16'})
axs[0].set_ylabel('Frequency',{'size':'14'})
plt.legend(loc=0)

caption = '''...'''
fig.text(.5,- .17, caption, ha='center',fontsize=14)

plt.show()
