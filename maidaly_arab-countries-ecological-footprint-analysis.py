# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import math
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
%matplotlib inline
sns.set(font_scale=1.1)
# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/NFA 2018.csv")
data.sample(10)
data.describe()
data.info()
def extract_country_by_record(df,country_name,record):
    country_foot_print=df[df.country.isin([country_name])]
    country_by_record = country_foot_print [country_foot_print.record.isin([record])]
    return country_by_record

def extract_countries_feature_by_year (df,countries_list,feature,year,record="BiocapPerCap"):
    excluded_countries=[]
    feature_values=[]
    available_countries=[]
    for i in range (0,len(countries_list)):
        country_by_record = extract_country_by_record(df,countries_list[i],record)
        feature_value = country_by_record.loc[lambda df1: country_by_record.year == year][feature].values
        if  feature_value.size==0 or math.isnan(feature_value[0]) :
            excluded_countries.append(countries_list[i])
        else:
            feature_values.append(feature_value[0]) 
            available_countries.append(countries_list[i])
            
#  activate if you need to print the excluded countries in the year
#     if len(excluded_countries) != 0:
#         print("excluded countries in {0} are : ".format(year))
#         for i in excluded_countries:
#             print(i)
    return feature_values, available_countries, excluded_countries 
def print_excluded_countries (excluded_countries,year):
    if len(excluded_countries) != 0:
        print("excluded countries from dataset in {0} are : ".format(year))
        for i in excluded_countries:
            print(i)   
            
def calculate_growth_rate(present,past,period):
    #present : present year , past: past year , period: number of years between present and past
    percentage_growth_rate = ((present - past)/(past*period))*100
    return percentage_growth_rate
    
arab_countries = ['Egypt','Algeria','Bahrain','Libyan Arab Jamahiriya',
                 'Jordan','Iraq','Mauritania','Morocco',
                  'Saudi Arabia','Kuwait','Qatar','Sudan (former)',
                 'Oman','Tunisia','United Arab Emirates','Yemen',
                  'Lebanon','Syrian Arab Republic','Somalia','Comoros','Djibouti']

colors = ['blue','xkcd:medium grey','red','green','pink',
          'xkcd:muted blue','yellow','magenta','brown',
          'orange','xkcd:tan','xkcd:seafoam','tab:olive',
          'xkcd:turquoise','xkcd:mauve','xkcd:acid green',
          'xkcd:bland','xkcd:coral','xkcd:chocolate','xkcd:red purple',
          'xkcd:bright lilac','xkcd:heather']

years=np.sort(data.year.unique())

arab_df = pd.DataFrame()
for country in arab_countries:
    arab_df=arab_df.append(data[data.country.isin([country])])

fig= plt.figure(figsize=(15,7))
regoin=[]
sub_region=[]
for country in arab_countries:
    regoin.append(arab_df[arab_df.country.isin([country])]["UN_region"].unique()[0])
    sub_region.append(arab_df[arab_df.country.isin([country])]["UN_subregion"].unique()[0])
plt.subplot2grid((1,2),(0,0))
sns.countplot(pd.Series(regoin))
plt.title("Arab countries distrbution according to regions")
plt.subplot2grid((1,2),(0,1))
sns.countplot(pd.Series(sub_region))
plt.title("Arab countries distrbution according to subregions")


fig = plt.figure(figsize=(15,20))
plt.subplot2grid((2,1),(0,0))
population,available_countries,excluded_countries=extract_countries_feature_by_year(arab_df,arab_countries,'population',2014)
population_df = pd.DataFrame({'countery':available_countries,'population':population}).sort_values(by='population',ascending=False)
population_list = list (population_df['population'])
# to avoid overlab of labels at the small slices in the chart the explode added 
# the explode len must be the same len of the pie data and adding excluded values to the equivalent pos of the required elements
# and the remaining elements left 0, the explode could be list or set
explode = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.3,0.5,0.7)
wedges, texts, autotexts = plt.pie(population_list, autopct= '%.1f%%',textprops=dict(color="black"),
                           colors= colors, labels= list(population_df['countery']),explode=explode,labeldistance =1.03)
# plt.legend(wedges, list(df['countery']),
#           title="Countries",
#           loc="center left",
#           bbox_to_anchor=(1, 0, 0.5, 1))
# plt.setp(autotexts, size=10, weight="bold")
plt.title("Population distribution in Arab countries(without Sudan)")
plt.subplot2grid((2,1),(1,0))
available_countries.append("Sudan")
population.append(37737900)
population_df = pd.DataFrame({'countery':available_countries,'population':population}).sort_values(by='population',ascending=False)
population_list = list (population_df['population'])
explode = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0.1,0.3,0.5,0.7)
wedges, texts, autotexts = plt.pie(population_list, autopct= '%1.1f%%',textprops=dict(color="black"),
                           colors= colors, labels= list(population_df['countery']),explode=explode,labeldistance =1.03)
plt.title("Population distribution in Arab countries (Sudan added)")
print_excluded_countries(excluded_countries,2014)

fig = plt.figure(figsize=(14,9))
ax = sns.barplot(population_df['population'],population_df['countery'], palette="Blues_d")
population_list = list (np.array(population_df['population'])/10**6)
list_counter = 0
# annotating the values
for p in ax.patches:        
    x = p.get_bbox().get_points()[:,0]
    y = p.get_bbox().get_points()[:,1]
    ax.annotate(str(population_list[list_counter])+" M" , (x.max()+500000, y.mean()), 
                horizontalalignment='left',
                verticalalignment='center')
    list_counter += 1
plt.ylabel("")
plt.title("Arab countries by population [2014]")

print ("The total Population of Arab countries in 2014 according to the available data and with adding Sudan is: {0}".format(np.sum(population_df['population'])))

plt.figure(figsize=(15,10))
for i in range(len(arab_countries)):
    country_by_record = extract_country_by_record(arab_df,arab_countries[i],'BiocapPerCap')
    sns.lineplot(country_by_record['year'],country_by_record['population'],
             label=arab_countries[i],
             color = colors[i])
    
# plt.gca().set_color_cycle(colors)
plt.legend()
plt.title("Population growth  in Arab Countries")
plt.show()
population_2000,available_countries,excluded_countries_2000=extract_countries_feature_by_year(arab_df,arab_countries,'population',2000)
population_2010,available_countries,excluded_countries_2010=extract_countries_feature_by_year(arab_df,arab_countries,'population',2010)
population_growth_rate = []
for i in range (0,len(population_2000)):
    growth_rate = calculate_growth_rate(population_2010[i],population_2000[i],10)
    population_growth_rate.append(growth_rate)
growth_rate_df = pd.DataFrame({"country":available_countries,"growth rate":population_growth_rate}).sort_values(by="growth rate",ascending=False)
print_excluded_countries(excluded_countries_2000,2000)  
print_excluded_countries(excluded_countries_2010,2010)
growth_rate_df
fig = plt.figure(figsize=(13,10))
ax = sns.barplot(growth_rate_df["growth rate"],growth_rate_df["country"],palette="rocket")
growth_rate_list = list(np.round(np.array(growth_rate_df["growth rate"]),2))
list_counter = 0
# annotating the values
for p in ax.patches:        
    x = p.get_bbox().get_points()[:,0]
    y = p.get_bbox().get_points()[:,1]
    ax.annotate(str(growth_rate_list[list_counter] )+ " %" , (x.max()+0.1, y.mean()), 
                horizontalalignment='left',
                verticalalignment='center',size=12)
    list_counter += 1
plt.xlabel("annual growth rate")
plt.ylabel("")
plt.title("Arab countries by annual population growth rate from [2000-2010]")

arab_countrs_population = []
for year in years:
    sum_population_per_year = np.array(extract_countries_feature_by_year(arab_df,arab_countries,'population',year)[0]).sum()
    arab_countrs_population.append(sum_population_per_year)

fig = plt.figure(figsize=(12,6))
# The period from 1985 to 2010
sns.lineplot(years[24:49],arab_countrs_population[24:49])
plt.xlabel("year")
plt.ylabel("population")
plt.title("Arab countries population growth from 1985 to 2010")
arab_population_growth_rate = calculate_growth_rate(arab_countrs_population[49],arab_countrs_population[24],25)
plt.text(1985,3.1*10**8,"growth rate = {0}%".format(np.round(arab_population_growth_rate,2)),size=15)

fig = plt.figure(figsize=(15,8))
GDP,available_countries, excluded_countries=extract_countries_feature_by_year(arab_df,arab_countries,'Percapita GDP (2010 USD)',2014)
GDP_df = pd.DataFrame({'country':available_countries,'GDP':GDP}).sort_values(by='GDP',ascending=False)
ax=sns.barplot(GDP_df['GDP'],GDP_df['country'],palette="Blues_d")
gdp_list = list (GDP_df['GDP'])
list_counter = 0
# annotating the values
for p in ax.patches:        
    x = p.get_bbox().get_points()[:,0]
    y = p.get_bbox().get_points()[:,1]
    ax.annotate(str(gdp_list[list_counter] )+ " $" , (x.max()+500, y.mean()), 
                horizontalalignment='left',
                verticalalignment='center')
    list_counter += 1
plt.xlabel("GDP per Capita")
plt.title("Arab Countries by GDP in 2014")
print_excluded_countries(excluded_countries,2014)

plt.figure(figsize=(15,8))
for i in range(len(arab_countries)):
    country_by_record = extract_country_by_record(arab_df,arab_countries[i],'BiocapPerCap')
    sns.lineplot(country_by_record['year'],
                 country_by_record['Percapita GDP (2010 USD)'], 
                 label=arab_countries[i],color = colors[i])
    y_text_label = country_by_record.loc[lambda df: country_by_record.year == 2014]['Percapita GDP (2010 USD)'].values 
    if arab_countries[i] in list(GDP_df.country[0:8]) and y_text_label.size!=0 and not math.isnan(y_text_label[0]) :
        plt.text(2015,y_text_label[0], arab_countries[i])

    if arab_countries[i] in list(GDP_df.country[0:6]) and y_text_label.size!=0 and not math.isnan(y_text_label[0]) :
        plt.text(2015,y_text_label[0], arab_countries[i])
World_by_record = extract_country_by_record(data,'World','BiocapPerCap')
ax = sns.lineplot(World_by_record['year'],
             World_by_record['Percapita GDP (2010 USD)'], 
             label='World',color = 'red',linewidth=3) 
ax.lines[18].set_linestyle("--")
plt.text(2015,World_by_record.loc[lambda df: World_by_record.year == 2014]['Percapita GDP (2010 USD)'].values [0],'World',color='red')
# plt.gca().set_color_cycle(colors)
plt.legend()
plt.title("Percapita GDP for Arab Contries")
mean_GDP = []
for year in years :
    mean_GDP.append(np.array(extract_countries_feature_by_year(arab_df,arab_countries,'Percapita GDP (2010 USD)',year)[0]).mean())
fig = plt.figure(figsize=(18,6))
plt.subplot2grid((1,2),(0,0))          # plot GDP from 1961 to 2014
sns.lineplot(years, mean_GDP)
plt.xlabel("year")
plt.ylabel("mean GDP value")
plt.title("Mean of the GDP from 1961 to 2014")
plt.subplot2grid((1,2),(0,1))           # plot GDP from 2000 to 2014 
sns.lineplot(years[39:], mean_GDP[39:])
plt.xlabel("year")
plt.ylabel("mean GDP value")
plt.title("Mean of the GDP from 2000 to 2014")

arab_consumption_corr=arab_df[arab_df.record.isin(["EFConsPerCap"])].drop('year',axis=1).corr()
fig=plt.figure(figsize=(10, 10))
sns.heatmap(arab_consumption_corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='Reds',linecolor="white",cbar=False,annot_kws={"size":12})
plt.title('Correlation between features according to ecological footprint (per person)')
biocapcity_corr=arab_df[arab_df.record.isin(["BiocapPerCap"])].drop('year',axis=1).corr()
fig=plt.figure(figsize=(10, 10))
sns.heatmap(biocapcity_corr, vmax=.8, linewidths=0.01,
            square=True,annot=True,cmap='Reds',linecolor="white",cbar=False,annot_kws={"size":12})
plt.title('Correlation between features according to biocapcity')
fig = plt.figure(figsize=(25,10))
plt.subplot2grid((1,2),(0,0))
BiocapPerCap,available_countries,excluded_countries=extract_countries_feature_by_year(arab_df,arab_countries,'total',2014)
BiocapPerCap_df = pd.DataFrame({'country':available_countries,'BiocapPerCap':BiocapPerCap}).sort_values(by='BiocapPerCap',ascending=False)
ax=sns.barplot(BiocapPerCap_df['BiocapPerCap'],BiocapPerCap_df['country'],palette="Blues_d")

plt.xlabel("Biocapacity per Capita")
plt.ylabel("")

plt.title("Arab Countries by Biocapacity per capita in 2014")

plt.subplot2grid((1,2),(0,1))
EFConsPerCap,available_countries,excluded_countries=extract_countries_feature_by_year(arab_df,arab_countries,'total',2014,record="EFConsPerCap")
EFConsPerCap_df = pd.DataFrame({'country':available_countries,'EFConsPerCap':EFConsPerCap}).sort_values(by='EFConsPerCap',ascending=False)
ax=sns.barplot(EFConsPerCap_df['EFConsPerCap'],EFConsPerCap_df['country'],palette="rocket")
plt.xlabel("EFConsPerCap")
plt.title("Arab Countries by Ecological Footprint per capita in 2014")
plt.ylabel("")
print_excluded_countries(excluded_countries,2014)
plt.figure(figsize=(15,10))
for i in range(len(arab_countries)):
    plt.plot(extract_country_by_record(arab_df,arab_countries[i],'BiocapPerCap')['year'],
             extract_country_by_record(arab_df,arab_countries[i],'BiocapPerCap')['total'], 
             label=arab_countries[i],
             color = colors[i])
    
# plt.gca().set_color_cycle(colors)
plt.legend()
plt.title("Arab Countries Biocapacity per capita")
plt.figure(figsize=(20,5))
plt.subplot2grid((1,2),(0,0))
plt.plot(extract_country_by_record(arab_df,'Egypt','BiocapPerCap')['year'],
         extract_country_by_record(arab_df,'Egypt','BiocapPerCap')['total'],
         color='green',
         label = 'BiocapTotGHA')
plt.plot(extract_country_by_record(arab_df,'Egypt','EFConsPerCap')['year'],
         extract_country_by_record(arab_df,'Egypt','EFConsPerCap')['total'],
         color='red',
         label = 'EFConsTotGHA')
plt.title(" Ecological Footprint vs Biocapacity for Egypt per capicta")
plt.subplot2grid((1,2),(0,1))
plt.plot(extract_country_by_record(arab_df,'Egypt','BiocapTotGHA')['year'],
         extract_country_by_record(arab_df,'Egypt','BiocapTotGHA')['total'],
         color='green',
         label = 'BiocapTotGHA')
plt.plot(extract_country_by_record(arab_df,'Egypt','EFConsTotGHA')['year'],
         extract_country_by_record(arab_df,'Egypt','EFConsTotGHA')['total'],
         color='red',
         label = 'EFConsTotGHA')    
# plt.gca().set_color_cycle(colors)
plt.legend()
plt.title(" Ecological Footprint vs Biocapacity for Egypt (gha)")
fig=plt.figure(figsize=(27,27))
count = 1 
record ={0:['BiocapPerCap','EFConsPerCap'],1:['BiocapTotGHA','EFConsTotGHA']}
for i in range (0,5):
#     for j in range (0,2):
    for r in record.keys():
        plt.subplot2grid((5,2),(i,r))
        plt.plot(extract_country_by_record(arab_df,arab_countries[count],record[r][0])['year'],
                     extract_country_by_record(arab_df,arab_countries[count],record[r][0])['total'],
                     color='green',
                     label = record[r][0])
        plt.plot(extract_country_by_record(arab_df,arab_countries[count],record[r][1])['year'],
                     extract_country_by_record(arab_df,arab_countries[count],record[r][1])['total'],
                     color='red',
                     label = record[r][1])
        plt.legend()
        if(r==0):
            plt.title("Ecological Footprint vs Biocapacity for {0} per capicta".format(arab_countries[count]))
        else:
            plt.title("Ecological Footprint vs Biocapacity for {0} (gha)".format(arab_countries[count]))
            
    count +=1
fig=plt.figure(figsize=(27,27))
for i in range (0,5):
#     for j in range (0,2):
    for r in record.keys():
        plt.subplot2grid((5,2),(i,r))
        plt.plot(extract_country_by_record(arab_df,arab_countries[count],record[r][0])['year'],
                     extract_country_by_record(arab_df,arab_countries[count],record[r][0])['total'],
                     color='red',
                     label = record[r][0])
        plt.plot(extract_country_by_record(arab_df,arab_countries[count],record[r][1])['year'],
                     extract_country_by_record(arab_df,arab_countries[count],record[r][1])['total'],
                     color='green',
                     label = record[r][1])
        plt.legend()
        if(r==0):
            plt.title("Ecological Footprint vs Biocapacity for {0} (gha per person)".format(arab_countries[count]))
        else:
            plt.title("Ecological Footprint vs Biocapacity for {0} (gha)".format(arab_countries[count]))
            
    count +=1
fig=plt.figure(figsize=(27,27))
for i in range (0,4):
    if count < len(arab_countries):
        for r in record.keys():
            plt.subplot2grid((4,2),(i,r))
            plt.plot(extract_country_by_record(arab_df,arab_countries[count],record[r][0])['year'],
                         extract_country_by_record(arab_df,arab_countries[count],record[r][0])['total'],
                         color='red',
                         label = record[r][0])
            plt.plot(extract_country_by_record(arab_df,arab_countries[count],record[r][1])['year'],
                         extract_country_by_record(arab_df,arab_countries[count],record[r][1])['total'],
                         color='green',
                         label = record[r][1])
            plt.legend()
            if(r==0):
                plt.title("Ecological Footprint vs Biocapacity for {0} (gha per person)".format(arab_countries[count]))
            else:
                plt.title("Ecological Footprint vs Biocapacity for {0} (gha)".format(arab_countries[count]))

        count +=1
fig=plt.figure(figsize=(27,27))
for i in range (0,4):
    if count < len(arab_countries):
        for r in record.keys():
            plt.subplot2grid((4,2),(i,r))
            plt.plot(extract_country_by_record(arab_df,arab_countries[count],record[r][0])['year'],
                         extract_country_by_record(arab_df,arab_countries[count],record[r][0])['total'],
                         color='red',
                         label = record[r][0])
            plt.plot(extract_country_by_record(arab_df,arab_countries[count],record[r][1])['year'],
                         extract_country_by_record(arab_df,arab_countries[count],record[r][1])['total'],
                         color='green',
                         label = record[r][1])
            plt.legend()
            if(r==0):
                plt.title("Ecological Footprint vs Biocapacity for {0} (gha per person)".format(arab_countries[count]))
            else:
                plt.title("Ecological Footprint vs Biocapacity for {0} (gha)".format(arab_countries[count]))

        count +=1
Arab_BiocapTotal = []
Arab_EFConsTotal = []
Arab_BiocapPerCap = []
Arab_EFConsPerCap = []
world_BiocapTotal = []
world_EFConsTotal = []
mean_BiocapPerCap = []
mean_EFConsPerCap = []
for year in years :
    sum_BiocapTotal_value = np.array(extract_countries_feature_by_year(arab_df,arab_countries,'total',year,record= 'BiocapTotGHA')[0]).sum()
    sum_EFConsTotal_value = np.array(extract_countries_feature_by_year(arab_df,arab_countries,'total',year,record='EFConsTotGHA')[0]).sum()
    sum_population_per_year = np.array(extract_countries_feature_by_year(arab_df,arab_countries,'population',year)[0]).sum()
    world_BiocapTotal.append(np.array(extract_countries_feature_by_year(data,['World'],'total',year,record= 'BiocapTotGHA')[0]))
    world_EFConsTotal.append(np.array(extract_countries_feature_by_year(data,['World'],'total',year,record= 'EFConsTotGHA')[0]))
    Arab_BiocapTotal.append(sum_BiocapTotal_value)
    Arab_EFConsTotal.append(sum_EFConsTotal_value)
    Arab_BiocapPerCap.append(sum_BiocapTotal_value/sum_population_per_year)
    Arab_EFConsPerCap.append(sum_EFConsTotal_value/sum_population_per_year)
    mean_BiocapPerCap.append(np.array(extract_countries_feature_by_year(arab_df,arab_countries,'total',year)[0]).mean())
    mean_EFConsPerCap.append(np.array(extract_countries_feature_by_year(arab_df,arab_countries,'total',year,record='EFConsPerCap')[0]).mean())
fig = plt.figure(figsize=(10,7))
sns.lineplot(years[19:], Arab_BiocapTotal[19:],color='green',label="BiocapTotGHA")
sns.lineplot(years[19:], Arab_EFConsTotal[19:],color='red',label="EFConsTotGHA")
plt.legend()
plt.xlabel("year")
plt.ylabel("GHA")
plt.title("The Ecological Footprint vs Biocapacity for Arab countries (gha)")

fig = plt.figure(figsize=(22,7))
fig.dpi=200
plt.subplot2grid((1,2),(0,0))
sns.lineplot(years[19:], Arab_BiocapPerCap[19:],color='green',label="BiocapPerCap")
sns.lineplot(years[19:], Arab_EFConsPerCap[19:],color='red',label="EFConsPerCap")
plt.legend()
plt.xlabel("year")
plt.ylabel("per person")
plt.title("The Ecological Footprint vs Biocapacity for Arab countries (per capita)")
plt.subplot2grid((1,2),(0,1))
sns.lineplot(years[19:], mean_BiocapPerCap[19:],color='green',label="BiocapPerCap")
sns.lineplot(years[19:], mean_EFConsPerCap[19:],color='red',label="EFConsPerCap")
plt.legend()
plt.xlabel("year")
plt.ylabel("mean value")
plt.title("Mean of the Ecological Footprint vs Biocapacity for Arab countries (per capita)")
# defict_res_2014 = mean_BiocapPerCap[-1] - mean_EFConsPerCap[-1]
# # if defict_res_2014 < 0 :
# #     print ("The arab countries in 2014 has an Ecological deficit by {0}".format(np.abs(defict_res_2014)))
# # else:
# #     print ("The arab countries in 2014 has an Ecological reserve by {0}".format(np.abs(defict_res_2014)))
difference  = []
countries_list = []
deficit_or_reserve = []
for country in arab_countries:
    BiocapPerCap=np.array(extract_countries_feature_by_year(arab_df,[country],'total',2014)[0])
    EFConsPerCap=np.array(extract_countries_feature_by_year(arab_df,[country],'total',2014,record="EFConsPerCap")[0])
    difference_value = BiocapPerCap - EFConsPerCap
    if difference_value < 0 :
        deficit_or_reserve.append ("deficit")
        difference.append(difference_value[0])
    if difference_value > 0 :
        deficit_or_reserve.append("reserve")
        difference.append(difference_value[0])
    if difference_value.size==0:
        deficit_or_reserve.append("nan")
        difference.append(np.NAN)
    countries_list.append(country)
defict_reserve_df = pd.DataFrame({"country":countries_list,"deficit/reserve":deficit_or_reserve,"value":difference}).dropna().sort_values(by="value",ascending=False)
defict_reserve_df
fig = plt.figure(figsize=(12,8))
sns.barplot(y=defict_reserve_df["country"], x=defict_reserve_df["value"],
                hue=defict_reserve_df["deficit/reserve"])
plt.ylabel("")
plt.title("Ecological Deficit/Reserve for Arab countries (per capita)")
import datetime
arab_eod_dates = []
eod_dates_world=[]
def calc_earth_overshot_day(biocap,ecofootp):
    eod = (np.array(biocap) / np.array(ecofootp))*365
    return eod
eod_arab = calc_earth_overshot_day(Arab_BiocapTotal,Arab_EFConsTotal)
eod_world = calc_earth_overshot_day(world_BiocapTotal,world_EFConsTotal)

for i in range (0,len(eod_arab)):
    if eod_arab[i]>365:
        arab_eod_dates.append("no EOD")
    if eod_world[i]>365:
        eod_dates_world.append("no EOD")
    if eod_arab[i] < 365:
        date_arab = datetime.datetime(years[i],1,1) + datetime.timedelta(days=eod_arab[i])
        arab_eod_dates.append(date_arab.strftime('%b-%d'))
    if eod_world[i] < 365:
        date_world = datetime.datetime(years[i],1,1) + datetime.timedelta(days=int(eod_world[i]))
        eod_dates_world.append(date_world.strftime('%b-%d'))
eod_df = pd.DataFrame({"year":years[19:]," Arab Earth Overshoot Day":arab_eod_dates[19:]," World Earth Overshoot Day":eod_dates_world[19:]})
eod_df
fig = plt.figure(figsize=(15,8))
Arab_carbon,available_countries,excluded_countries=extract_countries_feature_by_year(arab_df,arab_countries,'carbon',2014,record="EFConsPerCap")
carbon_df = pd.DataFrame({'country':available_countries,'carbon':Arab_carbon}).sort_values(by='carbon',ascending=False)
ax=sns.barplot(carbon_df['carbon'],carbon_df['country'],palette="rocket")
plt.xlabel("Footprint of Carbon (GHA/person)")
plt.ylabel("")
plt.title("Arab Countries by Footprint of Carbon (GHA/person) in 2014")
print_excluded_countries(excluded_countries,2014)
Arab_EFConsCarbonPerCap=[]
Arab_EFConsCarbonTot =[]
for i in range (len(years)):
    sum_EFConsCarbonTot_value = np.array(extract_countries_feature_by_year(arab_df,arab_countries,'carbon',years[i],record='EFConsTotGHA')[0]).sum()
    Arab_EFConsCarbonTot.append(sum_EFConsCarbonTot_value)
    Arab_EFConsCarbonPerCap.append(sum_EFConsCarbonTot_value/arab_countrs_population[i])
fig = plt.figure(figsize=(22,7))
fig.dpi=200
plt.subplot2grid((1,2),(0,0))
sns.lineplot(years[19:], Arab_EFConsCarbonTot[19:],color='green')
plt.xlabel("year")
plt.ylabel("Carbon Footprint (GHA)")
plt.title("The Ecological Footprint of Carbon for Arab Countries ")
plt.subplot2grid((1,2),(0,1))
sns.lineplot(years[19:], Arab_EFConsCarbonPerCap[19:],color='green')
plt.xlabel("year")
plt.ylabel("Carbon Footprint(GHA/person)")
plt.title("The Ecological Footprint of Carbon for Arab Countries(per capita)")
fig = plt.figure(figsize=(15,8))
Arab_crop_land,available_countries,excluded_countries=extract_countries_feature_by_year(arab_df,arab_countries,'crop_land',2014,record="EFConsPerCap")
crob_land_df = pd.DataFrame({'country':available_countries,'crop_land':Arab_crop_land}).sort_values(by='crop_land',ascending=False)
ax=sns.barplot(crob_land_df['crop_land'],crob_land_df['country'],palette="rocket")
plt.xlabel("Footprint of Crop Land (GHA/person)")
plt.ylabel("")
plt.title("Arab Countries by Footprint of Crop Land (GHA/person) in 2014")
print_excluded_countries(excluded_countries,2014)
Arab_EFConsCropLandPerCap=[]
Arab_EFConsCropLandTot =[]
for i in range (len(years)):
    sum_EFConsCropLandTot_value = np.array(extract_countries_feature_by_year(arab_df,arab_countries,'crop_land',years[i],record='EFConsTotGHA')[0]).sum()
    Arab_EFConsCropLandTot.append(sum_EFConsCropLandTot_value)
    Arab_EFConsCropLandPerCap.append(sum_EFConsCropLandTot_value/arab_countrs_population[i])
fig = plt.figure(figsize=(22,7))
fig.dpi=200
plt.subplot2grid((1,2),(0,0))
sns.lineplot(years[19:50], Arab_EFConsCropLandTot[19:50],color='red')       # years[19:50] the years from 1980 to 2010
sns.regplot(years[19:50], Arab_EFConsCropLandTot[19:50],color='green',ci=68,scatter_kws={"s": 0})
plt.xlabel("year")
plt.ylabel("Crop Land Footprint (GHA)")
plt.title("The Ecological Footprint of Crop Land for Arab Countries ")
plt.subplot2grid((1,2),(0,1))
sns.lineplot(years[19:50], Arab_EFConsCropLandPerCap[19:50],color='red')
sns.regplot(years[19:50], Arab_EFConsCropLandPerCap[19:50],color='green',ci=68,scatter_kws={"s": 0})
plt.xlabel("year")
plt.ylabel("Crop Land Footprint(GHA/person)")
plt.title("The Ecological Footprint of Crop Land for Arab Countries(per capita)")