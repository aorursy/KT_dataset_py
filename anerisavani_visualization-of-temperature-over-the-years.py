import pandas as pd
from pandas import DataFrame
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib import rcParams
import plotly.graph_objects as go
import plotly.express as px
from plotly.colors import n_colors
import numpy as np
import datetime as dt
import plotly.express as px
import seaborn as sns
#reading the file
GlobalTemp = pd.read_csv("/kaggle/input/climate-change-earth-surface-temperature-data/GlobalTemperatures.csv", parse_dates= ['dt'])
GlobalTempCountry = pd.read_csv("/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCountry.csv", parse_dates= ['dt'])
GlobalTempState = pd.read_csv("/kaggle/input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByState.csv", parse_dates= ['dt']) 

GlobalTemp.head(5)
GlobalTemp.rename(columns = {'dt':'Date'}, inplace = True)
print(GlobalTemp.Date.min())
print(GlobalTemp.Date.max())
GlobalTemp.dtypes
Year_Temp = GlobalTemp.groupby(GlobalTemp['Date'].dt.year)['LandAverageTemperature','LandMaxTemperature',
                                                           'LandMinTemperature','LandAndOceanAverageTemperature'].mean().reset_index()
Year_Temp.rename(columns = {'Date':'Year'}, inplace = True)

fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=Year_Temp.Year, y=Year_Temp.LandAverageTemperature,
                    mode='lines',
                    name='LandAvgTemp',
                    marker_color='#A9A9A9'))
fig.add_trace(go.Scatter(x=Year_Temp.Year, y=Year_Temp.LandMaxTemperature,
                    mode='lines',
                    name='LandMaxAvgTemp',
                    marker_color='#BDB76B'))
fig.add_trace(go.Scatter(x=Year_Temp.Year, y=Year_Temp.LandMinTemperature,
                    mode='lines',
                    name='LandMinAvgTemp',
                    marker_color='#45CE30'))

fig.add_trace(go.Scatter(x=Year_Temp.Year, y=Year_Temp.LandAndOceanAverageTemperature,
                    mode='lines',
                    name='Land&OceanAvgTemp',
                    marker_color='#FFA07A'))
fig.update_layout(
    height=800,
    xaxis_title="Years",
    yaxis_title='Temperatures in degree',
    title_text='Average Land, Ocean, Minimun, and Maximum Temperatures over the years'
)
fig.add_annotation(
            x=1950,
            y=2.7,
            text="1950")
fig.add_annotation(
            x=1972,
            y=8.4,
            text="1972")
fig.add_annotation(
            x=1978,
            y=14.28,
            text="1978")
fig.add_annotation(
            x=1969,
            y=15.31,
            text="1969")
fig.update_annotations(dict(
            xref="x",
            yref="y",
            showarrow=True,
            arrowhead=7,
            ax=0,
            ay=-40
))

fig.update_layout(showlegend=True)




fig.show()

GlobalTemp["Year"] = pd.DatetimeIndex(GlobalTemp['Date']).year
GlobalTemp["Month"] = pd.DatetimeIndex(GlobalTemp['Date']).month
GlobalTemp['Month'] = GlobalTemp['Month'].astype(str) 
GlobalTemp.loc[GlobalTemp['Month']=='1','Month'] = 'January'
GlobalTemp.loc[GlobalTemp['Month']=='2','Month'] = 'February'
GlobalTemp.loc[GlobalTemp['Month']=='3','Month'] = 'March'
GlobalTemp.loc[GlobalTemp['Month']=='4','Month'] = 'April'
GlobalTemp.loc[GlobalTemp['Month']=='5','Month'] = 'May'
GlobalTemp.loc[GlobalTemp['Month']=='6','Month'] = 'June'
GlobalTemp.loc[GlobalTemp['Month']=='7','Month'] = 'July'
GlobalTemp.loc[GlobalTemp['Month']=='8','Month'] = 'August'
GlobalTemp.loc[GlobalTemp['Month']=='9','Month'] = 'September'
GlobalTemp.loc[GlobalTemp['Month']=='10','Month'] = 'October'
GlobalTemp.loc[GlobalTemp['Month']=='11','Month'] = 'November'
GlobalTemp.loc[GlobalTemp['Month']=='12','Month'] = 'December'
year_month = GlobalTemp.groupby(by = ['Year','Month']).mean().reset_index()
# Figure size
plt.figure(figsize=(16,12))

# The plot
sns.boxplot(x = 'Month', y = 'LandAverageTemperature', data = year_month, palette = "RdBu", saturation = 1, width = 0.9, fliersize=4, linewidth=2)

# Make pretty
plt.title('Average Temperature on Land by Months', fontsize = 25)
plt.xlabel('Months', fontsize = 20)
plt.ylabel('Temperature', fontsize = 20)

month_temp = GlobalTemp.groupby(by = ['Year','Month']).mean().reset_index()

July = month_temp.loc[month_temp['Month'] == 'July',:]
August = month_temp.loc[month_temp['Month'] == 'August',:]
January = month_temp.loc[month_temp['Month'] == 'January',:]
February = month_temp.loc[month_temp['Month'] == 'February',:]
December = month_temp.loc[month_temp['Month'] == 'December',:]
fig1 = go.Figure()
for template in ["plotly_dark"]:
    fig1.add_trace(go.Scatter(x=July['Year'], y=July['LandAverageTemperature'],
                    mode='lines',
                    name='July',
                    marker_color='#f075c2'))
    fig1.add_trace(go.Scatter(x=August['Year'], y=August['LandAverageTemperature'],
                    mode='lines',
                    name='August',
                    marker_color='#28d2c2'))
    fig1.add_trace(go.Scatter(x=January['Year'], y=January['LandAverageTemperature'],
                    mode='lines',
                    name='January',
                    marker_color='#ffd201'))
    fig1.add_trace(go.Scatter(x=February['Year'], y=February['LandAverageTemperature'],
                    mode='lines',
                    name='February',
                    marker_color='#00C957'))
    fig1.add_trace(go.Scatter(x=December['Year'], y=December['LandAverageTemperature'],
                    mode='lines',
                    name='December',
                    marker_color='#F7F7F7'))
    fig1.update_layout(
    height=800,
    xaxis_title="Years",
    yaxis_title='Temperature in degree',
    title_text='Average Temperature in the months of July, August, January, and February over the years',
    template=template)



fig1.show()

month_season = {
    "January": "Winter",
    "February": "Winter",
    "March": "Spring",
    "April": "Spring",
    "May": "Spring",
    "June": "Summer",
    "July": "Summer",
    "August": "Summer",
    "September": "Autumn",
    "October": "Autumn",
    "November": "Autumn",
    "December": "Winter"
}

GlobalTemp['Season'] = ''

for month, season in month_season.items():
    GlobalTemp.loc[GlobalTemp['Month'] == month, 'Season'] = season

year_season = GlobalTemp.groupby(by = ['Year','Season']).mean().reset_index()

Winter = year_season.loc[year_season['Season'] == 'Winter',:]
Spring = year_season.loc[year_season['Season'] == 'Spring',:]
Summer = year_season.loc[year_season['Season'] == 'Summer',:]
Autumn = year_season.loc[year_season['Season'] == 'Autumn',:]

fig2 = go.Figure()
for template in ["plotly_white"]:
    fig2.add_trace(go.Scatter(x=Winter['Year'], y=Winter['LandAverageTemperature'],
                    mode='lines',
                    name='Winter',
                    marker_color='#838B8B'))
    fig2.add_trace(go.Scatter(x=Spring['Year'], y=Spring['LandAverageTemperature'],
                    mode='lines',
                    name='Spring',
                    marker_color='#FFB5C5'))
    fig2.add_trace(go.Scatter(x=Summer['Year'], y=Summer['LandAverageTemperature'],
                    mode='lines',
                    name='Summer',
                    marker_color='#87CEFF'))
    fig2.add_trace(go.Scatter(x=Autumn['Year'], y=Autumn['LandAverageTemperature'],
                    mode='lines',
                    name='Autumn',
                    marker_color='#FF8000'))
    fig2.update_layout(
    height=800,
    xaxis_title="Years",
    yaxis_title='Temperature in degree',
    title_text='Average Temperature seasonwise over the years',
    template=template)




fig2.show()


GlobalTempCountry.head(5)
GlobalTempCountry.dtypes
GlobalTempCountry.rename(columns = {'dt':'Date'}, inplace = True)
print(GlobalTempCountry.Date.min())
print(GlobalTempCountry.Date.max())
GlobalTempCountry.Country.unique()
GlobalTempCountry.Country.nunique()
country_temp = GlobalTempCountry.groupby(by = ['Country']).mean().reset_index()
fig3 = px.choropleth(country_temp, locations="Country", locationmode = "country names", color="AverageTemperature",
                    color_continuous_scale=px.colors.diverging.BrBG,
                    title="Average Temperature Contrywise Worldwide")
fig3.show()
country_temp_asc = GlobalTempCountry.groupby(by = ['Country']).mean().reset_index().sort_values('AverageTemperature',ascending=False).reset_index(drop=True)
sns.set(style="whitegrid",font_scale=0.9)

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(6, 50))

# Plot the temperature
sns.set_color_codes("pastel")
sns.barplot(x="AverageTemperature", y="Country", data=country_temp_asc,
            label="Temperature",palette="Greens_d")

# Informative axis label
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set(xlim=(-20, 40), ylabel="",
       xlabel="Average Temperature")
sns.despine(left=True, bottom=True)
GlobalTempCountry["Year"] = pd.DatetimeIndex(GlobalTempCountry['Date']).year
GlobalTempCountry["Month"] = pd.DatetimeIndex(GlobalTempCountry['Date']).month
year_country = GlobalTempCountry.groupby(by = ['Year', 'Country']).mean().reset_index()
Russia = year_country.loc[year_country['Country'] == 'Russia',:]
Greenland = year_country.loc[year_country['Country'] == 'Greenland',:]
Denmark = year_country.loc[year_country['Country'] == 'Denmark',:]
Djibouti = year_country.loc[year_country['Country'] == 'Djibouti',:]
Mali= year_country.loc[year_country['Country'] == 'Mali',:]
Norway= year_country.loc[year_country['Country'] == 'Norway',:]
fig4 = go.Figure()
for template in ["plotly_dark"]:
    fig4.add_trace(go.Scatter(x=Russia['Year'], y=Russia['AverageTemperature'],
                    mode='lines',
                    name='Russia',
                    marker_color='#00CD66'))
    fig4.add_trace(go.Scatter(x=Greenland['Year'], y=Greenland['AverageTemperature'],
                    mode='lines',
                    name='Greenland',
                    marker_color='#FF4040'))
    fig4.add_trace(go.Scatter(x=Denmark['Year'], y=Denmark['AverageTemperature'],
                    mode='lines',
                    name='Denmark',
                    marker_color='#FFFF00'))
    fig4.add_trace(go.Scatter(x=Mali['Year'], y=Mali['AverageTemperature'],
                    mode='lines',
                    name='Mali',
                    marker_color='#EE82EE'))
    fig4.add_trace(go.Scatter(x=Djibouti['Year'], y=Djibouti['AverageTemperature'],
                    mode='lines',
                    name='Djibouti',
                    marker_color='#98F5FF'))
    fig4.add_trace(go.Scatter(x=Norway['Year'], y=Norway['AverageTemperature'],
                    mode='lines',
                    name='Norway',
                    marker_color='#E9967A'))
    fig4.update_layout(
          height=800,
          xaxis_title="Years",
          yaxis_title='Temperature in degree',
          title_text='Average Temperature Over the Years for the Following Coutries',
          template=template)




fig4.show()

GlobalTempState.dtypes
GlobalTempState.head(5)
GlobalTempState.rename(columns = {'dt':'Date'}, inplace = True)
print(GlobalTempState.Date.min())
print(GlobalTempState.Date.max())
GlobalTempState.Country.unique()
GlobalTempState.State.unique()
GlobalTempState.State.nunique()
country_temp_asc = GlobalTempState.groupby(by=['Country']).mean().reset_index().sort_values('AverageTemperature',ascending=False).reset_index(drop=True)
country_temp_asc
plt.figure(figsize=(20,10))

fig5 = px.bar(country_temp_asc, x='Country', y='AverageTemperature',color='AverageTemperature')

fig5.update_layout(
        title="Average Temperature of Countries Over 270 Years ",
        xaxis_title="Years",
        yaxis_title="Average Temperature",
        font=dict(
            family="Courier New",
            size=18,
            color="black"
        )
    )
fig5.show()
country_state_temp = GlobalTempState.groupby(by = ['Country','State']).mean().reset_index().sort_values('AverageTemperature',ascending=False).reset_index()
country_state_temp
country_state_temp["world"] = "world" # in order to have a single root node
fig6 = px.treemap(country_state_temp.head(200), path=['world', 'Country','State'], values='AverageTemperature',
                  color='State',color_continuous_scale='RdBu')
fig6.show()
GlobalTempState["Year"] = pd.DatetimeIndex(GlobalTempState['Date']).year
year_state = GlobalTempState.groupby(by = ['Year', 'State']).mean().reset_index()
Puducherry = year_state.loc[year_state['State'] == 'Puducherry',:]
Tamil_Nadu = year_state.loc[year_state['State'] == 'Tamil Nadu',:]
Amazonas = year_state.loc[year_state['State'] == 'Amazonas',:]
Nunavut= year_state.loc[year_state['State'] == 'Nunavut',:]
Evenk= year_state.loc[year_state['State'] == 'Evenk',:]
Sakha= year_state.loc[year_state['State'] == 'Sakha',:]
fig7 = go.Figure()
for template in ["plotly_dark"]:
    fig7.add_trace(go.Scatter(x=Puducherry['Year'], y=Puducherry['AverageTemperature'],
                    mode='lines',
                    name='Puducherry',
                    marker_color='#00CD66'))
    fig7.add_trace(go.Scatter(x=Tamil_Nadu['Year'], y=Tamil_Nadu['AverageTemperature'],
                    mode='lines',
                    name='Tamil Nadu',
                    marker_color='#FF4040'))
    fig7.add_trace(go.Scatter(x=Amazonas['Year'], y=Amazonas['AverageTemperature'],
                    mode='lines',
                    name='Amazonas',
                    marker_color='#FCE6C9'))
    fig7.add_trace(go.Scatter(x=Nunavut['Year'], y=Nunavut['AverageTemperature'],
                    mode='lines',
                    name='Nunavut',
                    marker_color='#FFFF00'))
    fig7.add_trace(go.Scatter(x=Evenk['Year'], y=Evenk['AverageTemperature'],
                    mode='lines',
                    name='Evenk',
                    marker_color='#EE82EE'))
    fig7.add_trace(go.Scatter(x=Sakha['Year'], y=Sakha['AverageTemperature'],
                    mode='lines',
                    name='Sakha',
                    marker_color='#98F5FF'))
    fig7.update_layout(
          height=800,
          xaxis_title="Years",
          yaxis_title='Temperature in degree',
          title_text='Average Temperature Over the Years for the Following States',
          template=template)




fig7.show()
