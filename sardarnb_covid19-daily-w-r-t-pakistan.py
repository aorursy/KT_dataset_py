import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly_express as px
import plotly.graph_objects as go
url = r'https://github.com/CSSEGISandData/COVID-19/raw/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
df = pd.read_csv(url)
df.columns
df = df.melt(id_vars=["Province/State", "Country/Region",'Lat','Long'], 
        var_name="Date", 
        value_name="Value")
df = df.rename(columns={'Province/State':'province','Country/Region':'country','Date':'date'})

df.head()
df.date = pd.to_datetime(df.date)
px.line(df.loc[(df.country=='India') | (df.country=='Pakistan')], 'date','Value',color='country',color_discrete_sequence=['red','darkgreen'], template='plotly_white',
       title='Total Confirmed Covid19 Cases')
# px.line(df, 'date','Value',color='country', template='plotly_white',
#        title='Total Confirmed Covid19 Cases')
df.loc[(df.country=='US') | (df.country=='China')].drop(['province'],axis=1)
df = df.groupby(['country','date'])[['Value']].sum().reset_index()
df['days_since_100'] = np.nan
df['days_since_1'] = np.nan
for country_name in df.loc[(df.Value>0)].country.unique():    
    df_new1 = df.loc[(df.country==country_name) & (df.Value>0)].reset_index(drop=True).reset_index().rename(columns={'index': 'days_since1'})
    df.loc[(df.country==country_name) & (df.Value>0),'days_since_1'] = df_new1['days_since1'].values +1
for country_name in df.loc[(df.Value>100)].country.unique():
    df_new = df.loc[(df.country==country_name) & (df.Value>100)].reset_index(drop=True).reset_index().rename(columns={'index': 'days_since'})
    df.loc[(df.country==country_name) & (df.Value>100),'days_since_100'] = df_new['days_since'].values +1


df.date = pd.to_datetime(df.date)
# df.loc[(df.country=='US') & (df.Value>100)].reset_index(drop=True).index.values

df.loc[(df.country=='Pakistan') & (df.Value>100)]
px.line(df,'days_since_1','Value',color='country')
px.line((df.loc[(df.country=='US') | (df.country=='India')| (df.country=='Pakistan') | (df.country=='Italy') | (df.country=='Germany')]), 'days_since_100','Value',color='country',
       hover_data=['date'], labels={'entity':'Country', 'days since the 100th total confirmed case (days)': 'Days since the 100th total confirmed case (days)','total confirmed cases (cases)':'Total Confirmed Cases'},
       title='<b>Trend after the 100th case of Covid19 is confirmed. Data updated till: 3rd April.</b><br>The date on which the 100th case was confirmed is different for each country.<br>However, the tally is started on that day making it easier for a comparison of the rate of infection.',
       template='plotly_white',color_discrete_sequence=['lightblue','red','orange','darkgreen','violet'])
px.line((df.loc[df.country!='China']), 'days_since_100','Value',color='country',
       hover_data=['date'], labels={'entity':'Country', 'days since the 100th total confirmed case (days)': 'Days since the 100th total confirmed case (days)','total confirmed cases (cases)':'Total Confirmed Cases'},
       title='<b>Trend after the 100th case of Covid19 is confirmed. Data updated till: 3rd April.</b><br>The date on which the 100th case was confirmed is different for each country.<br>However, the tally is started on that day making it easier for a comparison of the rate of infection.',
       template='plotly_white',color_discrete_sequence=['lightblue','red','orange','darkgreen','violet'])
px.line((df.loc[df.country!='China']), 'days_since_1','Value',color='country',
       hover_data=['date'], labels={'entity':'Country', 'days since the 100th total confirmed case (days)': 'Days since the 100th total confirmed case (days)','total confirmed cases (cases)':'Total Confirmed Cases'},
       title='<b>Trend after the 100th case of Covid19 is confirmed. Data updated till: 3rd April.</b><br>The date on which the 100th case was confirmed is different for each country.<br>However, the tally is started on that day making it easier for a comparison of the rate of infection.',
       template='plotly_white',color_discrete_sequence=['lightblue','red','orange','darkgreen','violet'])

pak_good = df.loc[df['days_since_100']== df.loc[df.country=='Pakistan']['days_since_100'].max()].reset_index(drop=True)
# import plotly.graph_objects as go
# color_discrete_sequence=["indianred", "darkgreen", "lightskyblue", "goldenrod", "magnta"]
pak_good = pak_good.sort_values('Value',ascending=False)
pak_good = pak_good.reset_index(drop=True)
coloring=['darkslategrey',] * pak_good.country.nunique()
coloring[pak_good.loc[pak_good['country']=='Pakistan'].index.item()] = 'darkgreen'
coloring[pak_good.loc[pak_good['country']=='India'].index.item()] = 'red'
xday = df.loc[df.country=='Pakistan']['days_since_100'].max()
todaysdate = pd.datetime.now().strftime("%d/%m/%Y")
fig = go.Figure(data=[go.Bar(
    x=pak_good['country'],
    y=pak_good['Value'],
    marker_color=coloring # marker color can be a single color value or an iterable
)])
fig.update_layout(template='plotly_white',title_text=f"<b>{todaysdate} is Pakistan's {xday} day after the 100th confirmed case.</b><br>The rest of the world at this time were:")

double_time = df.loc[(df.country=='Pakistan') & (df['days_since_100'] >= 0)].reset_index(drop=True)

double_time['day'] = double_time.index 
double_time['2days'] = double_time['Value'] *2

double_time['2days'] = double_time['Value'][0] ** double_time.day

double_time['pct_change'] = double_time['Value'].pct_change()
x0 = double_time['Value'][0]

r = round(double_time['pct_change'].mean(),2)


double_time['2days'] = x0 * (1 + r)**double_time.day
fig = px.scatter(double_time,'day','Value',trendline='ols', color='country',
          template='plotly_white',title=f'Pakistan covid19 cases somehow maintaining a linear relationship.<br>OLS y=160x-188,where x=days after 100th confirmed case.Rsquared=0.94.<br>Transmission rate a little under {r*100}%.',
          labels={'entity':'Actual', 'days since the 100th total confirmed case (days)': 'Days since the 100th total confirmed case (days)','total confirmed cases (cases)':'Total Confirmed Cases'},
          hover_data=['date'],
          )
fig.add_trace(
    go.Scatter(
        x=double_time.day,
        y=double_time['2days'],
        mode="lines",
        line=go.scatter.Line(color="gray"), name=f"If rate of transmission is {r*100}%",
        showlegend=True)
)
fig.show()
double_time_1 = df.loc[(df.country=='Pakistan') & (df['days_since_1'] >= 0)].reset_index(drop=True)

double_time_1['day'] = double_time_1.index 

double_time_1['2days'] = double_time_1['Value'] *2

double_time_1['2days'] = double_time_1['Value'][0] ** double_time.day

double_time_1['pct_change'] = double_time_1['Value'].pct_change()

x0 = double_time_1['Value'][0]

r = round(double_time_1['pct_change'].mean(),2)


double_time_1['2days'] = x0 * (1 + r)**double_time_1.day

fig = px.scatter(double_time_1,'day','Value',trendline='ols', color='country',
          template='plotly_white',title=f'Pakistan covid19 cases somehow maintaining a strong linear relationship.<br>OLS y=142x-59.8,where x=days after 100th confirmed case.Rsquared=0.968.<br>Transmission rate a little under {r*100}%.',
          labels={'entity':'Actual', 'days since the 100th total confirmed case (days)': 'Days since the 100th total confirmed case (days)','total confirmed cases (cases)':'Total Confirmed Cases'},
          hover_data=['date'],
          )
fig.add_trace(
    go.Scatter(
        x=double_time_1.day,
        y=double_time_1['2days'],
        mode="lines",
        line=go.scatter.Line(color="gray"), name=f"If rate of transmission is {r*100}%",
        showlegend=True)
)
fig.show()
fig = px.bar(double_time,'date','pct_change',title="Percentage increase of Pakistan's confirmed Covid19 cases per day<br>Data Source: Johns Hopkins",
      template='plotly_white', labels={'pct_change':'Percentage Change','date':'Date'})

fig.update_layout(
    showlegend=False,
    annotations=[
        dict(
            x='2020-03-24',
            y=0.133,
            xref="x",
            yref="y",
            text="Lockdown announced",
            showarrow=True,
            arrowhead=1,
            ax=45,
            ay=-80
        )
    ]
)
# fig.show()

fig = px.bar(double_time_1,'date','pct_change',title="Percentage increase of Pakistan's confirmed Covid19 cases per day<br>Data Source: Johns Hopkins",
      template='plotly_white', labels={'pct_change':'Percentage Change','date':'Date'})

fig.update_layout(
    showlegend=False,
    annotations=[
        dict(
            x='2020-03-24',
            y=0.133,
            xref="x",
            yref="y",
            text="Lockdown announced",
            showarrow=True,
            arrowhead=1,
            ax=45,
            ay=-80
        )
    ]
)
# fig.show()
#---
df.head()
# fig = px.line((df.loc[((df.country=='India') | (df.country=='Pakistan')) & (df['days_since_100'].notna())]), 'days_since_100','Value', log_y=False, animation_frame='country', animation_group='country',color='country',
#      #  range_x=[1,100000], range_y=[100,10000],
#        labels={'entity':'Country', 'days since the 100th total confirmed case (days)': 'Days since the 100th total confirmed case (days)','total confirmed cases (cases)':'Total Confirmed Cases'},
#        title='<b>Trend after the 100th case of Covid19 is confirmed. Data updated till: 3rd April.</b><br>The date on which the 100th case was confirmed is different for each country.<br>However, the tally is started on that day making it easier for a comparison of the rate of infection.',
#        template='plotly_white',color_discrete_sequence=['lightblue','red','orange','darkgreen','violet'])

# fig.update_traces(mode='lines')
appended_data = []
for day in df['days_since_100'].unique():
    df_ap = df.loc[df['days_since_100'] <= day]
    df_ap['ddays'] = day
    # store DataFrame in list
    appended_data.append(df_ap)
# see pd.concat documentation for more info
appended_data = pd.concat(appended_data)
appended_data.loc[appended_data.country=='Pakistan']
df_anim = appended_data.loc[((appended_data.country=='Italy') | (appended_data.country=='Germany') | (appended_data.country=='US')| (appended_data.country=='China'))]
df_anim[df_anim.ddays <= df_anim.days_since_100.max()]
fig = px.line(df_anim[df_anim.ddays <= df_anim.days_since_100.max()], 'date','Value', log_y=True, animation_frame='ddays', animation_group='country',color='country',
       range_x=[df_anim.date.min(),df_anim.date.max()], range_y=[100,df_anim['Value'].max()],
       labels={'entity':'Country', 'days_since_100': 'Days since the 100th total confirmed case (days)','total confirmed cases (cases)':'Total Confirmed Cases','ddays':'Days since the 100th total confirmed case (days)','date':'Date','Value':'Count of confirmed cases'},
       title=f'<b>Trend after the 100th case of Covid19 is confirmed. Data updated on: {todaysdate}.</b><br>The date on which the 100th case was confirmed is different for each country.<br>',
       template='plotly_white',color_discrete_sequence=['red','blue','orange','darkgreen','violet'])

fig.update_traces(mode='lines')
fig = px.bar(df_anim[df_anim.ddays <= df_anim.days_since_100.max()], 'country','Value', log_y=False, animation_frame='ddays', animation_group='Value',color='country',
       #range_x=[df_anim.date.min(),df_anim.date.max()], 
       range_y=[100,df_anim['Value'].max()],
       labels={'entity':'Country', 'days_since_100': 'Days since the 100th total confirmed case (days)','total confirmed cases (cases)':'Total Confirmed Cases','ddays':'Days since the 100th total confirmed case (days)','date':'Date','Value':'Count of confirmed cases'},
       title=f'<b>Trend after the 100th case of Covid19 is confirmed. Data updated on: {todaysdate}.</b><br>The date on which the 100th case was confirmed is different for each country.<br>',
       template='plotly_white',color_discrete_sequence=['red','blue','orange','darkgreen','violet'])

fig.update_traces()
##---- curve fittinggg
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
 
#Fitting function
def func(x, a, b):
    return a*np.exp(b*x)
    #return a*x+b
 
def func_2(x,a,b):
    return a**(b*x)
df_pak = df.loc[(df.country=='Pakistan') & (df['days_since_100'])]
df_pak.head()
xData = np.array(df_pak['days_since_100'])
yData = np.array(df_pak['Value'])
 
#Plot experimental data points
plt.plot(xData, yData, 'bo', label='experimental-data')
 
# Initial guess for the parameters
initialGuess = [1.0,1.0]    
 
#Perform the curve-fit
popt, pcov = curve_fit(func, xData, yData,)
print(popt)
 
#x values for the fitted function
xFit = np.arange(df_pak['days_since_100'].min(),df_pak['days_since_100'].max() + 10, 1)
yFit = func(xFit, *popt)

#Plot the fitted function
plt.plot(xFit, func(xFit, *popt), 'r', label='fit params: a=%5.3f, b=%5.3f' % tuple(popt))
 
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
fig = px.line(xFit,yFit)

fig = go.Figure()

fig.add_trace(go.Scatter(x=xFit, y=yFit,
                    mode='markers+lines',
                    name='Predicted'))

fig.add_trace(go.Scatter(x=xData, y=yData,
                    mode='markers',
                    name='Actual'))

fig.update_layout(title=f'<b>Actual vs Predicated Covid19 cases for {df_pak.country.unique()}</b><br>{popt[0].round(2)} * e^({popt[1].round(3)} * x(days after 100th confirmed case))<br>R-squared={r2_score(yFit[:upto],yData).round(3)}',
                 template='plotly_white')

popt[1]
379*np.exp(.101*20)

print(f'{popt[0].round(2)} * e^({popt[1].round(3)} * x(days after 100th confirmed case))')
from sklearn.metrics import r2_score
upto = df_pak['days_since_100'].max() 
upto = int(upto)
print(r2_score(yFit[:upto],yData))
