# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
case_data=pd.read_csv("../input/coronaviruscovid19-canada/cases.csv")
mort_data=pd.read_csv("../input/coronaviruscovid19-canada/mortality.csv")
recov_data=pd.read_csv("../input/coronaviruscovid19-canada/recovered.csv")
test_data=pd.read_csv("../input/coronaviruscovid19-canada/testing.csv")
pro_data=case_data['province'].value_counts(ascending=False).to_frame()
pro_data=pro_data.reset_index()
pro_data.columns=['province', 'Cases']

sns.set(style="whitegrid")
ax = sns.barplot(x=pro_data.province, y=pro_data.Cases)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
case_data.replace('Not Reported', np.nan, inplace=True)
age_data=case_data['age'].value_counts(dropna=True, ascending=False).to_frame()
age_data=age_data.reset_index()
age_data.columns=['Age_Range', 'Cases']
age_data=age_data.drop(age_data.index[9:16])
age_data['Age_Range'].replace({"<20" :"0-20"}, inplace=True)
age_data=age_data.sort_values('Age_Range')
age_data['Percentage']=((age_data.Cases/age_data.Cases.sum())*100).round(1).astype(str)+"%"
labels=age_data.Age_Range
explode=[0,0,0,0.1,0.1,0,0,0,0,0,0,0,0,0,0]
color_list=['gold', 'yellowgreen', 'lightskyblue', 'lightgreen', 'pink', 'red', 'lightcoral', 'purple', 'cyan']
fig,ax1=plt.subplots(figsize=(24,12))
ax1.pie(age_data['Cases'],labels=labels, autopct='%.1f%%',
        explode=explode, shadow=True, colors=color_list)
plt.title('Percentage of Cases by Group Age')
ax1.legend(labels, loc="upper right")
pro_recov=recov_data.loc[(recov_data['date_recovered']=='2020-04-03')]
pro_recov=pro_recov.drop(['province_source', 'source','date_recovered'], axis=1).reset_index(drop=True)
pro_recov.dropna(inplace=True)
pro_recov=pro_recov.rename(columns={'cumulative_recovered':'recovered'})
death_data=mort_data['province'].value_counts(ascending=False).to_frame()
death_data.reset_index(inplace=True)
death_data=death_data.rename(columns={'index':'province', 'province':'death'})
#pro_recov['death']=pro_recov.merge(death_data, on="province")['death']
result=pd.merge(death_data, pro_recov, on='province')
Final_data=pro_data.merge(death_data.merge(pro_recov, on='province'), on='province')
pro_data.plot(kind="barh", figsize=(10,6), x='province', y='Cases')
plt.title('Number of Confirmed Cases in Provinces')
plt.xlabel('Number of Cases')
plt.ylabel('Provinces')
pro_test=test_data.loc[(test_data['date_testing']=='2020-04-03')]
pro_test=pro_test.drop(['date_testing', 'province_source', 'source'], axis=1)
pro_test=pro_test.rename(columns={'cumulative_testing':'tested'})
#Final_data.loc['Canada']=Final_data.sum()
#Final_data.reset_index(inplace=True, drop=True)
Final_data.loc['Canada']=Final_data.select_dtypes(pd.np.number).sum()
Final_data.reset_index(inplace=True, drop=True)
Final_data.fillna('Canada', inplace=True)
Final_data.set_index('province', inplace=True)
#pie_data=Final_data.loc[['BC', 'Quebec','Ontario', 'Alberta']]
#color=['lightblue','r','lightgreen']
#label=['Active', 'Death', 'Cured']
#pie_data.T.plot.pie(subplots=True, figsize=(24,12),
                  #  colors=color, layout=(2,2),
                  #    legend=False, labels=label, autopct='%.1f%%',
                 #  title='Number of Active, Cured & Death Cases in Selected Provinces')
####Trend of Covid 19
trend_data=case_data['date_report'].value_counts(ascending=True).to_frame().reset_index()
trend_data=trend_data.rename(columns={'index':'Date_reported', 'date_report':'Number_of_Cases'})


trend_data.plot(kind="line", figsize=(10,6), x='Date_reported', y='Number_of_Cases', linewidth=2.5, color='maroon')
plt.title('Number of Coronvirus Positive Cases Over Time')
plt.xlabel('Day')
plt.ylabel('Number of Cases')

death_trend=mort_data['date_death_report'].value_counts(ascending=True).to_frame().reset_index()
death_trend=death_trend.rename(columns={'index':'Date_reported', 'date_death_report':'Number_of_death'})
death_trend.plot(kind='line', figsize=(10,6), x='Date_reported', y='Number_of_death', linewidth=2.5, color='coral')
plt.title('Number of Death Over Time')
plt.xlabel('Day')
plt.ylabel('Number of Death')

case_data.head()
case_map=case_data.drop(['age', 'sex', 'health_region', 'country', 'travel_yn', 'travel_history_country','locally_acquired','case_source', 'additional_info', 'additional_source','method_note','case_id', 'provincial_case_id','report_week'], axis=1)
case_map=case_map.rename(columns={'province':'Provinces'})
case_map.head(20)
map_final=case_map.groupby(['date_report'])['Provinces'].value_counts().to_frame()

map_final=map_final.rename(columns={'Provinces':'Confirmed'})
map_final.head()
map_final1=map_final.reset_index()
map_final1.head()
Lon_data={'Ontario':-85.000000, 'BC':-127.647621,'Quebec':-70.000000,'Alberta':-115.000000,'Saskatchewan': -106.000000,
         'Nova Scotia':63.000000,'NL':-60.000000, 'Manitoba':-98.813873, 'New Brunswick':-66.159668, 'PEI':-63.000000,
         'Yukon':-135.000000, 'NWT':-114.371788}

Lat_data={'Ontario':50.000000, 'BC':53.726669,'Quebec':53.000000,'Alberta':55.000000,'Saskatchewan':55.000000,
         'Nova Scotia':45.000000,'NL':53.000000, 'Manitoba':53.760860, 'New Brunswick':46.498390, 'PEI':46.250000,
         'Yukon':64.000000, 'NWT':69.445358}
map_final1['Lon']=map_final1['Provinces'].map(Lon_data)
map_final1['Lat']=map_final1['Provinces'].map(Lat_data)

map_final1.head()
import plotly.express as px
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)
#map_final1['date_report'] = map_final1['date_report'].dt.strftime('%Y/%m/%d')
fig = px.scatter_geo(map_final1,lat="Lat", lon="Lon", color='Confirmed', size='Confirmed', 
                     projection="natural earth",
                     hover_name="Provinces", scope='north america', animation_frame="date_report", 
                     range_color=[0, max(map_final1['Confirmed'])], title='Tend of Covid 19 through the time')
#fig.update_layout(plot_bgcolor='rgb(275, 270, 273)')

fig.show()
map_final1.to_csv('covid_19_canada.csv')

pro_data['Lon']=pro_data['province'].map(Lon_data)
pro_data['Lat']=pro_data['province'].map(Lat_data)
pro_data.dropna(inplace=True)
pro_data=pro_data.reset_index(drop=True)
pro_data.head(20)

import folium
map=folium.Map(location=[55.585901, -105.750596], zoom_start=6,max_zoom=4,min_zoom=3, 
                   tiles = "CartoDB dark_matter",height = 800,width = '100%')
for i in range (0, len(pro_data)):
    folium.Circle(
        location=[pro_data.iloc[i]['Lat'], pro_data.iloc[i]['Lon']],
        popup=pro_data.iloc[i]['province'],
        radius=pro_data.iloc[i]['Cases']*10,
         tooltip= "<h5 style='text-align:center;font-weight: bold'>"+pro_data.iloc[i].province+"</h5>"+
                    "<hr style='margin:10px;'>"+
                   "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+
       "<li>Cases: "+str(pro_data.iloc[i]['Cases'])+"</li>",
        color='crimson',
        fill=True,
        fill_color='green').add_to(map)
pro_data['Cases']=pro_data.Cases.astype(float)    
map.save('mymap.html')
map

