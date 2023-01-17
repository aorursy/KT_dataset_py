import datetime

import numpy as np

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

%matplotlib inline

import matplotlib.pyplot as plt

from matplotlib.pyplot import cm

import matplotlib.ticker as ticker

import matplotlib.animation as animation

from scipy.optimize import curve_fit

import os



# Input data files are available in the '/kaggle/input' or '../../../datasets/extracts/' directory.

file_input=['/kaggle/input','../../../datasets/extracts/']

files={}

for dirname, _, filenames in os.walk(file_input[0]):

    for filename in filenames:

        print(dirname,filename)

        if 'csv' in filename:

            files[filename.replace('.csv','')]=os.path.join(dirname, filename)

            print(filename.replace('.csv',''))
IndiaDF = pd.read_csv(files['covid_19_india'],

                      converters={

                          'ConfirmedIndianNational':lambda row: int(row.replace('-','0')),

                          'ConfirmedForeignNational':lambda row: int(row.replace('-','0')),

                          'Deaths':lambda row: int(''.join(filter(str.isdigit,row))) if row!='' else 0

                      })

IndiaDF = IndiaDF.rename({'State/UnionTerritory':'State','Cured':'Recovered'},axis=1)

IndiaDF['Active'] = IndiaDF['Confirmed'] - ( IndiaDF['Recovered']+ IndiaDF['Deaths'])

IndiaDF.head()
IndiaDF['Date'] = pd.to_datetime(IndiaDF['Date'],format='%d/%m/%y')

IndiaDF.sort_values('Date',inplace=True)

IndiaTotalDF = IndiaDF[['Date','Recovered','Deaths','Active','Confirmed']].groupby('Date').sum().reset_index()



fig = px.line(pd.melt(IndiaTotalDF,id_vars=['Date'], var_name='Value Type', value_name='Count'),

            x = 'Date',

            y = 'Count',

            color = 'Value Type',

            line_shape='spline',

            template='plotly_dark',

            title='Reported cases in India over Time')

fig.update_layout(yaxis={'type':'linear'})

fig.show()
columns=['Active','Recovered','Deaths']

IndiaPercDF=IndiaTotalDF.set_index('Date')[columns]

IndiaPercDF=IndiaPercDF.div(IndiaPercDF.sum(axis=1), axis=0).multiply(100)

IndiaPercDF.reset_index(inplace=True)



fig=go.Figure(data=go.Pie(labels=columns,

                values=[IndiaPercDF.iloc[IndiaPercDF['Date'].idxmax(axis=1)]['Active'],

                        IndiaPercDF.iloc[IndiaPercDF['Date'].idxmax(axis=1)]['Recovered'],

                       IndiaPercDF.iloc[IndiaPercDF['Date'].idxmax(axis=1)]['Deaths']

                       ]),layout={'template':'plotly_dark'})

fig.update_layout(title_text="Coronavirus Cases in India a/o "+IndiaTotalDF['Date'].max().strftime("%d-%b'%y"))

fig.show()



meltedDF=pd.melt(IndiaPercDF[columns[::-1]+['Date']],id_vars=['Date'], var_name='Value Type', value_name='Share Percentage')

fig = px.bar(meltedDF, 

       x = "Share Percentage",

       animation_frame = meltedDF['Date'].astype(str), 

       color = 'Value Type', 

       barmode = 'stack', height=400,

       template='plotly_dark',

       title='Cases percentage share over time',

       orientation='h')

fig.show()
DeltaColumns = ['Date','Confirmed', 'Recovered', 'Deaths']

fig = px.line(pd.melt((IndiaTotalDF[DeltaColumns].set_index('Date').diff()).reset_index(),id_vars=['Date'], var_name='Value Type', value_name='Count'),

            x = 'Date',

            y = 'Count',line_shape='spline',

            color = 'Value Type',

            template = 'plotly_dark',

            title='Cases per Day')

fig.update_layout(yaxis={'type':'linear'})

fig.show()



fig = px.line(pd.melt((IndiaTotalDF[DeltaColumns].set_index('Date').ewm(span=14).mean().diff()).reset_index(),id_vars=['Date'], var_name='Value Type', value_name='Count'),

            x = 'Date',

            y = 'Count',line_shape='spline',

            color = 'Value Type',

            template = 'plotly_dark',

            title='Exponential Weight 14 days Mean Cases per Day')

fig.update_layout(yaxis={'type':'linear'})

fig.show()



fig = px.line(pd.melt((IndiaTotalDF[DeltaColumns].set_index('Date').pct_change().ewm(span=14).mean()).reset_index(),id_vars=['Date'], var_name='Value Type', value_name='Count'),

            x = 'Date',

            y = 'Count',line_shape='spline',

            color = 'Value Type',

            template = 'plotly_dark',  

            title='Percentage Change per day with EWM 14 days mean')

fig.update_layout(yaxis={'type':'linear'})

fig.show()
IndianStatesDF = IndiaDF.sort_values(['State','Date']).drop_duplicates('State', keep='last')[['State','Recovered','Deaths','Confirmed','Active']]

IndianStatesDF['State'] = IndianStatesDF['State'].str.replace('#','')

IndianStatesDF = IndianStatesDF.groupby('State').sum().reset_index()



for item in columns:

    fig = px.treemap(IndianStatesDF,

                     path = ['State'],

                     values = item,

                     color = item,

                     title = item+' cases on different states',

                     template = 'plotly_dark')

    fig.show()
# Converting absolute values into percentage share

IndiaStateDF = IndiaDF.set_index(['Date','State'])[['Recovered','Deaths','Active']]

IndiaStateDF = IndiaStateDF.div(IndiaStateDF.sum(axis=1), axis=0).multiply(100)                          

IndiaStateDF.reset_index(inplace=True)

IndiaStateDF.sort_values('Date',inplace=True)



# Ranking the starting point of each States as 1st day for the the respective state

IndiaStateRankDF = pd.concat([IndiaStateDF,

           IndiaStateDF.groupby('State')['Date'].rank("dense",ascending=True).rename('Days')],

          axis=1)



# Bar chart race for showing contries infection rate over time (showing 10 ten at a time)

plt.rcParams["animation.html"] = "jshtml"

fig = plt.figure(figsize=(15, 8))

ax = fig.add_subplot()

confirmed_col = 'Recovered'



colors = dict(zip(

    IndiaStateRankDF['State'].unique(),

    cm.rainbow(np.linspace(0,1,len(IndiaStateRankDF['State'].unique()))

)))



def draw_barchart(current_year):

    dff = IndiaStateRankDF[(IndiaStateRankDF['Date'].eq(current_year))].sort_values(by=[confirmed_col,'State'], ascending=True).tail(10).fillna(0)

    

    ax.clear()

    ax.barh(dff['State'], dff[confirmed_col], color=[colors[x] for x in dff['State']])

    dx = dff[confirmed_col].max() / 100

    

    for i, (value, name) in enumerate(zip(dff[confirmed_col], dff['State'])):

        ax.text(value-dx, i,     name,             size=14, weight=600, ha='right', va='bottom')

        ax.text(value+dx, i,     f'{value:,.0f}',  size=14, ha='left',  va='center')



    ax.text(1, 0, current_year, transform=ax.transAxes, color='#777777', size=46, ha='right', weight=800)

    ax.text(0, 1.06, confirmed_col, transform=ax.transAxes, size=12, color='#777777')

    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'+'%'))

    ax.xaxis.set_ticks_position('top')

    ax.tick_params(axis='x', colors='#777777', labelsize=12)

    ax.set_yticks([])

    ax.margins(0, 0.01)

    

    ax.grid(which='major', axis='x', linestyle='-')

    ax.set_axisbelow(True)

    ax.text(0, 1.15, 'Recovered Rate(%) over Time',

            transform=ax.transAxes, size=24, weight=600, ha='left', va='top')

    ax.text(1, -0.1, 'by @Arnab Majumdar', transform=ax.transAxes, color='#777777', ha='right',

            bbox=dict(facecolor='white', alpha=0.8, edgecolor='white'))

    plt.box(False)

    plt.close()

    

# animating each frame of matplotlib chart using Funcanimation

animator = animation.FuncAnimation(fig, 

                                   draw_barchart, 

                                   frames=pd.date_range(start=IndiaStateRankDF[IndiaStateRankDF['Recovered']>0]['Date'].min(),

                                                        end=IndiaStateRankDF['Date'].max(),

                                                        freq='D').strftime('%Y-%m-%d'),

                                                        repeat=False,

                                                        cache_frame_data=True,

                                    interval=500)

animator
ICMRdf = pd.read_csv(files['ICMRTestingLabs'])

ICMRdf['type'] = ICMRdf['type'].str.replace('Laboratory','Lab')



fig = px.treemap(ICMRdf.groupby(['state','city','type'])['lab'].count().rename('count').reset_index(),

                 path=['state','city','type'],

                 values='count',

                 color='count',

                 title='ICMR Testing Labs in India',

                 height=700,

                 template='plotly_dark')

fig.update_layout(annotations= [{

    'text': "Click on State name/City name to deep dive",

      'font': {

      'size': 13,

      'color': 'rgb(256, 256, 256)',

    },

    'showarrow': False,

    'align': 'center',

    'x': 0.5,

    'y': 1,

    'xref': 'paper',

    'yref': 'paper',

  }])

fig.show()
ICMRTestingDF = pd.read_csv(files['StatewiseTestingDetails'])

ICMRTestingDF['Date'] = pd.to_datetime(ICMRTestingDF['Date'],format='%Y-%m-%d')

ICMRTestingDF = ICMRTestingDF[ICMRTestingDF['Date']<pd.to_datetime('today')]

ICMRTestingDF.sort_values('Date',inplace=True)

ICMRTestingDF = ICMRTestingDF.groupby('Date').sum().reset_index()

ICMRTestingDF['Cases/Total Tested Ratio'] = ICMRTestingDF['Positive']*100/ICMRTestingDF['TotalSamples']



fig = px.line(ICMRTestingDF,

              x = 'Date',

              y = 'Cases/Total Tested Ratio',

              template = 'plotly_dark',

              title = 'Positive cases/Total Tested Ratio')

fig.show()
IndivDF = pd.read_csv(files['IndividualDetails'])

IndivDF['diagnosed_date'] = pd.to_datetime(IndivDF['diagnosed_date'],format='%d/%m/%Y')

IndivDF['status_change_date'] = pd.to_datetime(IndivDF['status_change_date'],format='%d/%m/%Y')

IndivDF['StatusChangeDays'] = (IndivDF['status_change_date'] - IndivDF['diagnosed_date']).dt.days



# remove negative records, because it can never be negative, so they must be either wrong information or 

# recontacted disease, since we dont know, what is the reason, we will drop them

IndivDF = IndivDF[IndivDF['StatusChangeDays']>=0]

IndivDF['age'] = IndivDF['age'].str.split('-',expand=True)[0].astype(float)



IndivDF.head()
# since recovered days cannot be 0(unless patient is tested on recovery/retested on recovery)

fig = px.histogram(IndivDF[(IndivDF['current_status']=='Recovered') & (IndivDF['StatusChangeDays']>0)].fillna('N/A'),

             x = 'StatusChangeDays', 

             color = 'gender',

             marginal = "box",

             template = 'plotly_dark',

             title = 'Recovery Time Distribution'

            )

fig.show()
fig = px.box(IndivDF.fillna('N/A'), 

             x = "gender",

             y = "age",

             color = 'gender',

             points = 'all',

             template = 'plotly_dark',

             title = 'Age/Gender share of confirmed cases')

fig.show()
highConfStates = IndivDF.groupby(['detected_state'])['id'].count().nlargest(10).index



fig = px.box(IndivDF[IndivDF['detected_state'].isin(highConfStates)].fillna('N/A'), 

             x = "gender",

             y = "age",

             color = 'gender',

             points = 'all',

             facet_col='detected_state', facet_col_wrap=2,

             template = 'plotly_dark',height=1600,

             title = 'Age/Gender distribution for most infected States')

fig.show()
popCensusDF = pd.read_csv(files['population_india_census2011'])

popCensusDF['Density'] = popCensusDF['Density'].str.split('/km2',expand=True)[0].str.replace(',','').astype(float)

popCensusDF['Area'] = popCensusDF['Area'].str.split('\xa0',expand=True)[0].str.replace(',','').astype(float)



fig = go.Figure(data=[

    go.Bar(name='Rural Population', 

           x=popCensusDF['State / Union Territory'].str.slice(0,15), 

           y=popCensusDF['Rural population']),

    go.Bar(name='Urban Population',  

           x=popCensusDF['State / Union Territory'].str.slice(0,15), 

           y=popCensusDF['Urban population'])

])



fig.update_layout(barmode='stack',

                  xaxis=dict(ticks="inside",tickangle = 45),

                  template='plotly_dark',

                 title='Population Share')

fig.show()
IndianStatesConDF = pd.merge(IndianStatesDF,popCensusDF,left_on='State',right_on='State / Union Territory', how='left')

IndianStatesConDF.drop(['Sno','State / Union Territory'],axis=1,inplace=True)



# Creating correlation matrix

corrMatrix = IndianStatesConDF.corr()

corrMatrix.style.background_gradient('Blues')
columnsWConf = columns + ['Confirmed']

for column in columnsWConf:

    possCorr = ','.join([item for item in corrMatrix[corrMatrix[column]>0].sort_values(column,ascending=False)\

                .index.tolist() if item not in columnsWConf])

    negCorr = ','.join([item for item in corrMatrix[corrMatrix[column]<0].sort_values(column,ascending=False)\

                .index.tolist() if item not in columnsWConf])

    print('\033[4m\033[1m\033[36m'+column+' cases\033[0m')

    print('Features with positive correlation: \033[1m\033[91m'+possCorr+'\033[0m')

    print('Features with negative correlation: \033[1m\033[91m'+negCorr+'\033[0m')

    print('-')

WorldDF = pd.read_csv(files['covid_19_data'],usecols=['ObservationDate','Province/State','Country/Region','Confirmed','Deaths','Recovered'])

WorldDF = WorldDF[(WorldDF['Country/Region']=='Mainland China')]

WorldDF = WorldDF.groupby(['ObservationDate']).sum().reset_index()

WorldDF['ObservationDate'] = pd.to_datetime(WorldDF['ObservationDate'])

WorldDF.sort_values('ObservationDate',inplace=True)



fig = go.Figure()

fig.add_trace(go.Scatter(

              x = WorldDF['ObservationDate'],

              y = WorldDF['Confirmed']))

fig.add_trace(go.Scatter(

              x = WorldDF['ObservationDate'],

              y = WorldDF.rolling('3D',on='ObservationDate')['Confirmed'].mean()))



fig.update_layout(template = 'plotly_dark',

                  title = 'Confirmed Cases in China over Time')

fig.show()
predIndiaDF = IndiaTotalDF.reset_index()

extended_period = 500



def sigmoid(x, L, k, x0):

    return L / (1 + np.exp(-k * (x - x0))) + 1



popt, pcov = curve_fit(sigmoid,  (predIndiaDF.index+1).astype(float), predIndiaDF['Confirmed'],  p0=(0,0,0) )



x0 = int(popt[2])

print('\033[1mx0 (point/day of inflexion):\033[0m',int(popt[2]))

print('\033[1mL (Maximum no.of cases):\033[0m',int(popt[0]) )

print('\033[1mk (Growth Rate):\033[0m',round(float(popt[1]),2) )

print('\033[1mPCOV: \033[0m\n',pcov )



fig = go.Figure()

dateRange = pd.date_range(pd.to_datetime(predIndiaDF['Date'].min()),\

                                         pd.to_datetime(predIndiaDF['Date'].min())+pd.DateOffset(extended_period))



fig.add_trace(go.Scatter(x = predIndiaDF['Date'],

                         y = predIndiaDF['Confirmed'],

                         mode = 'lines',

                         name = 'Observed'))

fig.add_trace(go.Scatter(x = dateRange,

                         y = sigmoid([x for x in range(extended_period)],*popt),

                         mode = 'lines',

                         name = 'Predicted'))

fig.add_trace(go.Scatter(x = [dateRange[x0], dateRange[x0]],

                         y = [0,  sigmoid([x for x in range(extended_period)],*popt)[x0]],

                         name = 'X0 - Inflexion point',

                         mode = 'lines'))

fig.update_layout(template='plotly_dark',title='Projected Confirmed Cases')

fig.show()
fx = sigmoid([x for x in range(250)],*popt)

ApproxPeak = np.argmin(fx<(0.9999*fx.max()))



# ApproxPeak

print('Date of reaching Approx Peak: \033[1m',\

      (pd.to_datetime(predIndiaDF['Date'].min())+pd.DateOffset(ApproxPeak)).strftime("%d-%b'%Y"))

print('\033[0mConfirmed cases on reaching Approx Peak: \033[1m',int(0.999*popt[0]))