import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

from matplotlib import pyplot as plt

%matplotlib inline

df_original = pd.read_csv('../input/countries-covid19jan-to-apr/train.csv')

df_original
for x in df_original.columns:

    print(x)
df_original.describe()
df_original.info()
Total_Values_count = []

for x in df_original.count():

    Total_Values_count.append(x)



Null_values_percentage = []

for x in (df_original.isnull().sum()/len(df_original)*100):

    Null_values_percentage.append(x)



Null_values_count = []    

for x in df_original.isnull().sum():

    Null_values_count.append(x)
Stats_df = pd.DataFrame()

Stats_df['Columns'] = df_original.columns

Stats_df['Null_Value_%'] = Null_values_percentage

Stats_df['Total_Values_count'] = Total_Values_count

Stats_df['Total_Null_Values_Count'] = Null_values_count

sns.set(rc={'figure.figsize':(15.7,8.27)})

sns.barplot(x=Stats_df['Columns'],y=Stats_df['Total_Values_count'])
sns.barplot(x=Stats_df['Columns'],y=Stats_df['Total_Null_Values_Count'])
sns.set(rc={'figure.figsize':(15.7,8.27)})

sns.barplot(x=Stats_df['Columns'],y=Stats_df['Null_Value_%'])
df_original.dtypes

df_original['Date'] = pd.to_datetime(df_original['Date'])

df_original.info()
df_original

df_temp_1 = df_original[['Date','ConfirmedCases','Fatalities']]
month_dict = {1:'January' , 2:'Februrary',3:'March',4:'April',5:'May',6:'June',7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}

df_temp_1['Month'] = pd.DatetimeIndex(df_temp_1['Date']).month

df_temp_1['Month'] = df_temp_1['Month'].map(month_dict) 

df_temp_1.drop('Date',axis=1)
months_list = df_temp_1['Month'].unique()

#sns.lineplot('Month', 'value', hue='variable', data=pd.melt(df_temp_1, 'Month'))
#Just wait a 1 to 2 min to let graphs display

df_temp_2 = pd.DataFrame()

for x in months_list:

    df_temp_2 = df_temp_1[df_temp_1['Month']==x]

    plt.figure()

    sns.lineplot(x=df_temp_2['ConfirmedCases'],y=df_temp_2['Fatalities']).set_title(x,size=30)

df_temp_2 = df_temp_2.drop(df_temp_2.index, inplace=True)

for x in months_list:

    df_temp_2 = df_temp_1[df_temp_1['Month']==x]

    plt.figure()

    sns.heatmap(df_temp_2.corr()).set_title(x,size=30)
df_temp_2.drop(df_temp_2.index,inplace=True)

deaths = []

cases = []

for x in months_list:

    df_temp_2 = df_temp_1[df_temp_1['Month']==x]

    deaths.append(df_temp_2['Fatalities'].values.sum())

    cases.append(df_temp_2['ConfirmedCases'].values.sum())



nd_array_1 = np.array(deaths)

nd_array_2 = np.array(cases)



nd_array_3 = (nd_array_1/nd_array_2)*100

nd_array_3

nd_array_3 = np.round(nd_array_3, 2)

death_ratio_value = list(nd_array_3)



deathRatio_info = {'Month':months_list,'Confirmed_Cases':cases,'Deaths':deaths,'Death_Ratio(%)':death_ratio_value}

deathRatio_DF = pd.DataFrame(deathRatio_info)



sns.barplot(x=months_list,y=death_ratio_value).set_title("Death Ratio %",size=30)
deathRatio_DF.set_index('Month')[['Confirmed_Cases', 'Deaths']].plot(kind='bar', figsize=(14, 10))

plt.xticks(rotation=60)

plt.title("Confirmed Cases & Deaths", fontsize=18, y=1.01)

plt.xlabel("Month", labelpad=15)

plt.ylabel("Value", labelpad=15)

plt.legend(["Confirmed Cases", "Deaths"], fontsize=16, title="Type");
sns.pairplot(df_temp_1)
countries = []

Confirmed_Cases = []

Deaths = []

for x in df_original['Country_Region'].unique():

    countries.append(x)

    k_1 = df_original[df_original['Country_Region']==x]

    Confirmed_Cases.append(k_1['ConfirmedCases'].sum())

    Deaths.append(k_1['Fatalities'].sum())



nd_array_1 = np.array(Confirmed_Cases)

nd_array_2 = np.array(Deaths)



nd_array_3 = (nd_array_2/nd_array_1)*100

nd_array_3 = np.round(nd_array_3,2)

Death_Ratio = list(nd_array_3)

Recovery_Chances = 100-nd_array_3

countries_df = pd.DataFrame({'Countries':countries,'ConfirmedCases':Confirmed_Cases,'Deaths':Deaths,'Death_Ratio':Death_Ratio,'Recovery_Chances':Recovery_Chances})

countries_df
import pandasql as ps

countries_cases = countries_df[['Countries', 'ConfirmedCases']]



Query1 = "SELECT Countries,SUM(ConfirmedCases) FROM countries_cases GROUP BY Countries;"



result_query1 = ps.sqldf(Query1, locals())

result_query1 = result_query1.loc[(result_query1 != 0).any(1)]

result_query1.sort_values(by=['SUM(ConfirmedCases)'], inplace=True)



plt.title("Countries with Confirmed Cases",bbox={'facecolor': '0.4', 'pad': 10})

x = np.char.array(result_query1['Countries'].tolist())

y = np.array(result_query1['SUM(ConfirmedCases)'].tolist())

percent = 100.*y/y.sum()



patches, texts = plt.pie(y, startangle=90, radius=1.2)

labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]



sort_legend = True

if sort_legend:

    patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),

                                          key=lambda x: x[2],

                                          reverse=True))



plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.5, 1.5),

           fontsize=8)

fig = plt.figure(figsize=[80, 80])


countries_cases = countries_df[['Countries', 'Deaths']]



Query1 = "SELECT Countries,SUM(Deaths) FROM countries_cases GROUP BY Countries;"



result_query1 = ps.sqldf(Query1, locals())

result_query1 = result_query1.loc[(result_query1 != 0).any(1)]

result_query1.sort_values(by=['SUM(Deaths)'], inplace=True)



plt.title("Countries with Deaths",bbox={'facecolor': '0.4', 'pad': 10})

x = np.char.array(result_query1['Countries'].tolist())

y = np.array(result_query1['SUM(Deaths)'].tolist())

percent = 100.*y/y.sum()



patches, texts = plt.pie(y, startangle=90, radius=1.2)

labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]



sort_legend = True

if sort_legend:

    patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),

                                          key=lambda x: x[2],

                                          reverse=True))



plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.5, 1.5),

           fontsize=8)

fig = plt.figure(figsize=[80, 80])
countries_cases = countries_df[['Countries', 'Death_Ratio']]



Query1 = "SELECT Countries,SUM(Death_Ratio) FROM countries_cases GROUP BY Countries;"



result_query1 = ps.sqldf(Query1, locals())

result_query1 = result_query1.loc[(result_query1 != 0).any(1)]

result_query1.sort_values(by=['SUM(Death_Ratio)'], inplace=True)



plt.title("Countries with Death_Ratio(%)",bbox={'facecolor': '0.4', 'pad': 10})

x = np.char.array(result_query1['Countries'].tolist())

y = np.array(result_query1['SUM(Death_Ratio)'].tolist())

percent = 100.*y/y.sum()



patches, texts = plt.pie(y, startangle=90, radius=1.2)

labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, percent)]



sort_legend = True

if sort_legend:

    patches, labels, dummy =  zip(*sorted(zip(patches, labels, y),

                                          key=lambda x: x[2],

                                          reverse=True))



plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.5, 1.5),

           fontsize=8)

fig = plt.figure(figsize=[80, 80])
df_top_10_CC = countries_df.nlargest(10,['ConfirmedCases'])
df_top_10_CC = df_top_10_CC[['Countries','ConfirmedCases']]

sns.barplot(x='Countries',y='ConfirmedCases',data=df_top_10_CC)
df_top_10_deaths = countries_df.nlargest(10,['Deaths'])

df_top_10_deaths = df_top_10_deaths[['Countries','Deaths']]

sns.barplot(x='Countries',y='Deaths',data=df_top_10_deaths)
df_top_10_dR = countries_df.nlargest(10,['Death_Ratio'])

df_top_10_dR = df_top_10_dR[['Countries','Death_Ratio']]

sns.barplot(x='Countries',y='Death_Ratio',data=df_top_10_dR)







df_top_10_RC = countries_df.nlargest(10,['Recovery_Chances'])

df_top_10_RC = df_top_10_RC[['Countries','Recovery_Chances']]

sns.barplot(x='Countries',y='Recovery_Chances',data=df_top_10_RC)


df_top_10_CC = countries_df.nsmallest(10,['ConfirmedCases'])

df_top_10_CC = df_top_10_CC[['Countries','ConfirmedCases']]

sns.set(rc={'figure.figsize':(20.7,8.27)})

sns.barplot(x='ConfirmedCases',y='Countries',data=df_top_10_CC)
df_top_10_dR = countries_df.nsmallest(10,['Death_Ratio'])

df_top_10_dR = df_top_10_dR[['Countries','Death_Ratio']]

sns.barplot(x='Death_Ratio',y='Countries',data=df_top_10_dR)
df_top_10_dR = countries_df.nsmallest(10,['Death_Ratio'])

df_top_10_dR = df_top_10_dR[['Countries','Death_Ratio']]

sns.barplot(x='Death_Ratio',y='Countries',data=df_top_10_dR)
df_top_10_RC = countries_df.nsmallest(10,['Recovery_Chances'])

df_top_10_RC = df_top_10_RC[['Countries','Recovery_Chances']]

sns.barplot(x='Recovery_Chances',y='Countries',data=df_top_10_RC)
df_top_20_CC = countries_df.nlargest(20,['ConfirmedCases'])

df_top_20_CC = df_top_20_CC[['Countries']]

names_countries = df_top_20_CC['Countries'].values.tolist()

for x in names_countries:

    data = df_original[df_original['Country_Region']==x]

    plt.figure()

    sns.jointplot(x=data['ConfirmedCases'],y=data['Fatalities'],kind='scatter')

    plt.title(x,size=30)

    

df_top_20_CC = countries_df.nlargest(20,['ConfirmedCases'])

df_top_20_CC = df_top_20_CC[['Countries']]

names_countries = df_top_20_CC['Countries'].values.tolist()

for x in names_countries:

    data = df_original[df_original['Country_Region']==x]

    y = data['ConfirmedCases'].values

    k = data['Fatalities'].values

    z = (k/y)*100

    z = np.round(z,2)

    data['Death_Ratio']=z

    plt.figure()

    sns.jointplot(x=data['ConfirmedCases'],y=data['Death_Ratio'],kind='scatter',color='orange')

    plt.title(x,size=30)