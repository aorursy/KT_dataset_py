!pip install calmap
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

import plotly.express as px
import plotly.offline as ply
import plotly.graph_objs as go
ply.init_notebook_mode(connected=True)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
from sklearn.linear_model import LinearRegression

import calmap
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
majordir='/kaggle/input/covid19-us-county-trend/'
datadir=majordir+'csse_covid_19_daily_reports/'
start_date='03-22-2020'
date_today='04-07-2020'
days=(pd.to_datetime(date_today)-pd.to_datetime(start_date)).days
days
# 48201.0
# aa=covid_data_world_daily_start_date[(covid_data_world_daily_start_date['Country_Region']=='US')]
# aa[(aa['FIPS']==49057.0)]
# covid_data_world_daily_start_date[(covid_data_world_daily_start_date['Province_State']=='New Hampshire')&(covid_data_world_daily_start_date['Admin2']=='Hillsborough')]
covid_data_world_daily_start_date=pd.read_csv(datadir+start_date+'.csv')
covid_data_us_daily_start_date=covid_data_world_daily_start_date[(covid_data_world_daily_start_date['Country_Region']=='US')][['FIPS','Lat','Long_','Confirmed']].copy()
covid_data_us_daily_start_date.rename(columns={'Confirmed':start_date},inplace=True)
covid_data_us_daily_start_date.head()
covid_data_us_daily_start_date= covid_data_us_daily_start_date[covid_data_us_daily_start_date['FIPS'].notna()]
vc=covid_data_us_daily_start_date['FIPS'].value_counts()
vclist=vc[vc > 1].index.tolist()
covid_data_us_daily_start_date=covid_data_us_daily_start_date[~(covid_data_us_daily_start_date['FIPS'].isin(vclist)&(covid_data_us_daily_start_date[start_date]>0))]
covid_data_us_daily_start_date.shape
# covid_data_us_daily_start_date[(covid_data_us_daily_start_date['FIPS']==33011.0)]
for i in range(days):
    date=(pd.to_datetime(start_date)+pd.DateOffset(i+1)).date().isoformat()
    date_reframe=date[5:7]+'-'+date[8:]+'-'+date[0:4]
    dataset=datadir+date_reframe+'.csv'
    covid_data_world_daily=pd.read_csv(dataset)
    covid_data_world_daily.rename(columns={'Confirmed':date_reframe},inplace=True)    
    covid_data_us_daily=covid_data_world_daily[covid_data_world_daily['Country_Region']=='US'].copy() 
#     print(covid_data_us_daily[(covid_data_us_daily['FIPS']==35013.0)][date_reframe])
    if i==0:
        print(i)
        covid_data_us_dailytrend=covid_data_us_daily_start_date[['FIPS','Lat','Long_',start_date]].merge(covid_data_us_daily[['FIPS',date_reframe]],on='FIPS').dropna()
    else:
        print(i)
        covid_data_us_dailytrend=covid_data_us_dailytrend.merge(covid_data_us_daily[['FIPS',date_reframe]],on='FIPS').dropna()
#     print(covid_data_us_dailytrend[(covid_data_us_dailytrend['FIPS']==33011.0)])
covid_data_us_dailytrend.shape
covid_data_us_dailytrend[(covid_data_us_dailytrend['FIPS']==33011.0)]
covid_data_us_dailytrend=covid_data_us_dailytrend.drop_duplicates(['FIPS'])
vc=covid_data_us_dailytrend['FIPS'].value_counts()
vclist=vc[vc > 1].index.tolist()
vc[vc > 1]
covid_data_us_dailytrend.sort_values(by=date_today,ascending=False).head()
covid_data_us_dailytrend[(covid_data_us_dailytrend['FIPS']==33011.0)]
census_df_fips = pd.read_excel(majordir+'PopulationEstimates_us_county_level_2018.xlsx',skiprows=1)
census_df_fips.FIPS=census_df_fips.FIPS.astype(float)
census_density_df_fips = pd.read_csv(majordir+'uscounty_populationdesity.csv', encoding = "ISO-8859-1",skiprows=1)
census_density_df_fips.rename(columns={'Target Geo Id2':'FIPS'},inplace=True)
census_pop_density_df_fips=census_df_fips.merge(census_density_df_fips[['FIPS','Density per square mile of land area - Population']],on='FIPS')
census_pop_density_df_fips.shape
census_pop_density_fips_covid=census_pop_density_df_fips.merge(covid_data_us_dailytrend,on='FIPS')
census_pop_density_fips_covid.sort_values(by=date_today,ascending=False).head()
census_pop_density_fips_covid.loc[census_pop_density_fips_covid.FIPS==36061,'POP_ESTIMATE_2018']=8398748
census_pop_density_fips_covid.sort_values(by=date_today,ascending=False).head()
census_pop_density_fips_covid_st200=census_pop_density_fips_covid[(census_pop_density_fips_covid[start_date]>200)].sort_values(by=date_today,ascending=False)
census_pop_density_fips_covid_st200['Combined_Key']=census_pop_density_fips_covid_st200['Area_Name'] +', '+ census_pop_density_fips_covid_st200['State']
# census_pop_density_fips_covid_st200[['State','Area_Name',start_date,date_today,'POP_ESTIMATE_2018']]

census_pop_density_fips_covid_st200[['State','Area_Name',start_date,date_today,'POP_ESTIMATE_2018']].style.background_gradient(cmap='Blues',subset=["03-22-2020"])\
                        .background_gradient(cmap='Reds',subset=["04-07-2020"])\
                        .background_gradient(cmap='Greens',subset=["POP_ESTIMATE_2018"]) 
census_pop_density_fips_covid_st200.head()
# census_pop_density_fips_covid_st200.T.iloc[5:-1].tail()
census_pop_density_fips_covid_st200_timeseries=census_pop_density_fips_covid_st200.T.iloc[7:-1]
census_pop_density_fips_covid_st200_timeseries.index=pd.to_datetime(census_pop_density_fips_covid_st200_timeseries.index)
census_pop_density_fips_covid_st200_timeseries.index
f = plt.figure(figsize=(20,10))
census_pop_density_fips_covid_st200_timeseries[1241]
calmap.yearplot(census_pop_density_fips_covid_st200_timeseries[1241], fillcolor='white', cmap='Blues', linewidth=0.5,linecolor="#fafafa",year=2020,)
plt.title("Daily Confirmed Cases",fontsize=20)
plt.tick_params(labelsize=15)
census_pop_density_fips_covid_st200_timeseries.columns=census_pop_density_fips_covid_st200.Combined_Key
census_pop_density_fips_covid_st200_timeseries.T
# Thanks to Tarun Kumar: https://www.kaggle.com/tarunkr/covid-19-case-study-analysis-viz-comparisons
temp = census_pop_density_fips_covid_st200_timeseries.T.copy()
threshold = 1000
f = plt.figure(figsize=(10,12))
ax = f.add_subplot(111)
for i,Combined_Key in enumerate(temp.index):
#     x = 30
    t = temp.loc[temp.index== Combined_Key].values[0]
    t = t[t>threshold]
     
    date = np.arange(0,len(t))
    xnew = np.linspace(date.min(), date.max(), 30)
    spl = make_interp_spline(date, t, k=1)  # type: BSpline
    power_smooth = spl(xnew)
    if Combined_Key[-2:] != 'NY':
        plt.plot(xnew,power_smooth,'-o',label = Combined_Key,linewidth =3, markevery=[-1])
#     else:
#         marker_style = dict(linewidth=4, linestyle='-', marker='o',markersize=10, markerfacecolor='#ffffff')
#         plt.plot(date,t,"-.",label = country,**marker_style)

plt.tick_params(labelsize = 14)        
plt.xticks(np.arange(0,30,7),[ "Day "+str(i) for i in range(30)][::7])     

# Reference lines 
# x = np.arange(0,18)
# y = 2**(x+np.log2(threshold))
# plt.plot(x,y,"--",linewidth =2,color = "gray")
# plt.annotate("No. of cases doubles every day",(x[-2],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)

# x = np.arange(0,26)
# y = 2**(x/2+np.log2(threshold))
# plt.plot(x,y,"--",linewidth =2,color = "gray")
# plt.annotate(".. every second day",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)

x = np.arange(0,26)
y = 2**(x/7+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "gray")
plt.annotate(".. every week",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)

x = np.arange(0,26)
y = 2**(x/30+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "gray")
plt.annotate(".. every month",(x[-3],y[-1]),xycoords="data",fontsize=14,alpha = 0.5)


x = np.arange(0,26)
y = 2**(x/4+np.log2(threshold))
plt.plot(x,y,"--",linewidth =2,color = "Red")
plt.annotate(".. every 4 days",(x[-3],y[-1]),color="Red",xycoords="data",fontsize=14,alpha = 0.8)

# plot Params
plt.xlabel("Days",fontsize=17)
plt.ylabel("Number of Confirmed Cases",fontsize=17)
plt.title("Trend Comparison of Different Counties (confirmed) (except for counties in New York State)",fontsize=22)
plt.legend(loc = "upper left")
plt.yscale("log")
plt.grid(which="both")
plt.show()
fig = px.scatter(census_pop_density_fips_covid_st200,x=start_date,y=date_today,hover_data=["Combined_Key"],color='State',size='POP_ESTIMATE_2018',size_max=40)
fig.update_layout(xaxis_type="log", yaxis_type="log")
fig.show()
census_pop_density_fips_covid_200=census_pop_density_fips_covid[(census_pop_density_fips_covid[date_today]>500)&(census_pop_density_fips_covid[date_today]<2000)].sort_values(by=date_today,ascending=False)
census_pop_density_fips_covid_200['Combined_Key']=census_pop_density_fips_covid_200['Area_Name'] +', '+ census_pop_density_fips_covid_200['State']
census_pop_density_fips_covid_200[['State','Area_Name','POP_ESTIMATE_2018',start_date,date_today,'Combined_Key']]
X = census_pop_density_fips_covid_200[start_date].values.reshape(-1, 1)  # values converts it into a numpy array
Y = census_pop_density_fips_covid_200[date_today].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression
Y_pred = linear_regressor.predict(X)  # make predictions
# census_pop_density_fips_covid_200[census_pop_density_fips_covid_200['Combined_Key'].isin(['Snohomish County, WA'])]
X1 = census_pop_density_fips_covid_200[census_pop_density_fips_covid_200['Combined_Key']!='Snohomish County, WA'][start_date].values.reshape(-1, 1)  # values converts it into a numpy array
Y1 = census_pop_density_fips_covid_200[census_pop_density_fips_covid_200['Combined_Key']!='Snohomish County, WA'][date_today].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression()  # create object for the class
linear_regressor.fit(X1, Y1)  # perform linear regression
Y_pred1 = linear_regressor.predict(X1)  # make predictions
fig = px.scatter(census_pop_density_fips_covid_200,x=start_date,y=date_today,color='State',size='POP_ESTIMATE_2018',size_max=40, hover_data=["Area_Name"])
# fig.add_scatter(x=X, y=Y_pred, mode='lines')
fig.add_traces(go.Scatter(x=census_pop_density_fips_covid_200[start_date].tolist(), y=pd.DataFrame(Y_pred)[0].tolist(),
                          mode = 'lines',
                          marker_color='black',
                          name='with Snohomish County')
                          )
fig.add_traces(go.Scatter(x=census_pop_density_fips_covid_200[census_pop_density_fips_covid_200['Combined_Key']!='Snohomish County, WA'][start_date].tolist(), y=pd.DataFrame(Y_pred1)[0].tolist(),
                          mode = 'lines',
                          marker_color='red',
                          name='without Snohomish County')
                          )
fig.show()
census_pop_density_fips_covid['Combined_Key']=census_pop_density_fips_covid['Area_Name'] +', '+ census_pop_density_fips_covid['State']
census_pop_density_fips_covid.head()
county_series=pd.DataFrame(census_pop_density_fips_covid[census_pop_density_fips_covid.columns[7:]].set_index('Combined_Key').unstack().reset_index()).merge(census_pop_density_fips_covid[['Combined_Key','Lat','Long_']],on='Combined_Key')
county_series.columns=['Date','county','Confirmed','Latitude','Longitude']
# county_series=county_series[county_series['Confirmed']>0].copy()
county_series['Confirmed_log']=np.log2(county_series['Confirmed'])
county_series.loc[county_series['Confirmed_log']==np.float('-inf'),'Confirmed_log']=0.0
county_series.sort_values(by='Date').head()
import plotly.express as px
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)


# tmp = df.copy()
# tmp['Date'] = tmp['Date'].dt.strftime('%Y/%m/%d')
fig = px.scatter_geo(county_series,lat="Latitude", lon="Longitude", color='Confirmed_log', size='Confirmed_log', locationmode = 'USA-states',
                     hover_name="county",hover_data=["Confirmed"], scope='usa', animation_frame="Date",
                     color_continuous_scale=px.colors.diverging.curl,center={'lat':39.50, 'lon':-98.35}, 
                     range_color=[0, max(county_series['Confirmed_log'])])
fig.update_layout(plot_bgcolor='rgb(275, 270, 273)')
fig.show()
pop_lakh_covid=census_pop_density_fips_covid[(census_pop_density_fips_covid['POP_ESTIMATE_2018']>100000)]['FIPS'].tolist()
pop_lakh_df=census_pop_density_df_fips[census_pop_density_df_fips['POP_ESTIMATE_2018']>100000]
# ST=['NY','PR']
# pop_lakh_df[(~pop_lakh_df.FIPS.isin(pop_lakh_covid) )& (~pop_lakh_df.State.isin(ST) ) ]
# census_pop_density_df_fips[(census_pop_density_df_fips['State']=='NH')]
# census_pop_density_fips_covid[(census_pop_density_fips_covid['State']=='NH')]
