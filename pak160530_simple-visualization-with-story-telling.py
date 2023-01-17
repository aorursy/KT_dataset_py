#Importing all libraries like numpy,plotly,seaborn,pandas,matplotlib,geocoder, time
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
########################################Set your plotly credentials#############################################################

import cufflinks as cf
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline
######################################## Run this only if you have google API key #############################################
#from pygeocoder import Geocoder
import time
plotly.offline.init_notebook_mode(connected=True)
##################################set the path to your directory where the csv files reside###################################
import os
print(os.listdir("../input/congressional-election-expenditures"))
#Importing data for first 3 visualizations
a = pd.read_csv('../input/congressional-election-expenditures/all_house_senate_2010.csv',usecols=[3,5,6])
b = pd.read_csv('../input/congressional-election-expenditures/2012.csv',usecols=[3,5,6])
c = pd.read_csv('../input/congressional-election-expenditures/all_house_senate_2014.csv',usecols=[3,5,6])
d = pd.read_csv('../input/congressional-election-expenditures/all_house_senate_2016.csv',usecols=[3,5,6],encoding = 'ISO-8859-1')
a.head()
#storing candidates by state in different dataframes
pieces = {'2010': a['can_nam'].groupby(a['can_off_sta']).nunique(),
          '2012': b['can_nam'].groupby(b['can_off_sta']).nunique(), 
          '2014': c['can_nam'].groupby(c['can_off_sta']).nunique(), 
          '2016': d['can_nam'].groupby(d['can_off_sta']).nunique()}

#merging results in a new dataframe
result = pd.concat(pieces, axis=1)
result1=result.reset_index()
results=result1.fillna(value=0)
results.head()
#Plot 1
#creating line chart (ploty package)
#defining lines
trace1 = go.Scatter(x =results['index'], y = results['2010'],mode = 'lines', name= 'cand. in 2010',opacity=0.9)
trace2 = go.Scatter(x =results['index'], y = results['2012'],mode = 'lines',name= 'cand. in 2012',opacity=0.8)
trace3 = go.Scatter(x =results['index'], y = results['2014'],mode = 'lines',name= 'cand. in 2014',opacity=0.9)
trace4 = go.Scatter(x =results['index'], y = results['2016'],mode = 'lines',name= 'cand. in 2016',opacity=0.7)

#defining graph properties
layout =go.Layout(title='Number of Participating Candidates(State Wise)',legend=dict(orientation="h",x=.2, y=1.0),
                  width=1000,
                  xaxis = dict(title = 'States'),
                  yaxis = dict(title = 'Number of Candidates'))
fig = go.Figure(data=[trace1,trace2,trace3,trace4], layout=layout)

#plotting the graph
plotly.offline.iplot(fig)
#storing candidates by office and storing it in new dataframe
pieces = {'2010': a['can_nam'].groupby(a['can_off']).nunique(), 
          '2012': b['can_nam'].groupby(b['can_off']).nunique(),
          '2014': c['can_nam'].groupby(c['can_off']).nunique(), 
          '2016': d['can_nam'].groupby(d['can_off']).nunique()}
result = pd.concat(pieces, axis=1)
result1=result.reset_index()

#filling with zeros for na values
results=result1.fillna(value=0)
results.head()
#Plot 2
#creating bar chart (ploty package)
#defining bar
trace1 = go.Bar(
    x=results['index'],
    y= results['2010'],
    name='cand. in 2010'
)
trace2 = go.Bar(
    x=results['index'],
    y= results['2012'],
    name='cand. in 2012'
)

trace3 = go.Bar(
    x=results['index'],
    y= results['2014'],
    name='cand. in 2014'
)

trace4 = go.Bar(
    x=results['index'],
    y= results['2016'],
    name='cand. in 2016'
)

data = [trace1,trace2,trace3,trace4]

#defining graph properties
layout = go.Layout(title='Distribution of Participating Candidates by offices',
                   height=650,
                   xaxis = dict(title = 'Offices'),
                   yaxis = dict(title = 'Number of Candidates', range=[0, 1900]),
                   barmode='group'
                    
)
fig = go.Figure(data=data, layout=layout)

#plotting graph
plotly.offline.iplot(fig)
# seperating dataframes by offices
oh2010=a.loc[a['can_off'] == 'H']
oh2012=b.loc[b['can_off'] == 'H']
oh2014=c.loc[c['can_off'] == 'H']
oh2016=d.loc[d['can_off'] == 'H']

os2010=a.loc[a['can_off'] == 'S']
os2012=b.loc[b['can_off'] == 'S']
os2014=c.loc[c['can_off'] == 'S']
os2016=d.loc[d['can_off'] == 'S']

#Counting the candidates by states in each dataframe
pieces = {'h2010': oh2010['can_nam'].groupby(oh2010['can_off_sta']).nunique(),
          'h2012': oh2012['can_nam'].groupby(oh2012['can_off_sta']).nunique(),
          'h2014': oh2014['can_nam'].groupby(oh2014['can_off_sta']).nunique(), 
          'h2016': oh2016['can_nam'].groupby(oh2016['can_off_sta']).nunique(),
          's2010': os2010['can_nam'].groupby(os2010['can_off_sta']).nunique(),
          's2012': os2012['can_nam'].groupby(os2012['can_off_sta']).nunique(),
          's2014': os2014['can_nam'].groupby(os2014['can_off_sta']).nunique(), 
          's2016': os2016['can_nam'].groupby(os2016['can_off_sta']).nunique()}

#merging results in a new dataframe
result = pd.concat(pieces, axis=1)

#filling with zeros for na values
result1=result.fillna(value=0)
results=result1.reset_index()
results.head()
#Plot 3
#creating scatter plot (ploty package)
trace1 = go.Scatter(x = results['index'], y = results['h2010'],mode = 'markers', name= 'H can. in 2010',
                    marker=dict(size= 11,opacity=0.8))
trace2 = go.Scatter(x =results['index'], y = results['h2012'],mode = 'markers', name= 'H can. in 2012',
                    marker=dict(size= 11,opacity=0.8))
trace3 = go.Scatter(x =results['index'], y = results['h2014'],mode = 'markers',name= 'H can. in 2014',
                    marker=dict(size= 11,opacity=0.8))
trace4 = go.Scatter(x =results['index'], y = results['h2016'],mode = 'markers', name= 'H can. in 2016',
                    marker=dict(size= 11,opacity=0.8))

#defining plot properties
layout =go.Layout(title='Distribution of House candidates by states',legend=dict(orientation="h"),
                  height=650,width=1050,
                  
                  xaxis = dict(title = 'States',showgrid=False,showline=True,zeroline=False,mirror="ticks",
                  ticks="inside",linewidth=2,tickwidth=2,zerolinewidth=2),
                  
                  yaxis = dict(title = 'Number of House Candidates',showgrid=True,showline=True,zeroline=False,
                  mirror="ticks",ticks="inside",linewidth=2,tickwidth=2,zerolinewidth=2))

fig = go.Figure(data=[trace1,trace2,trace3,trace4], layout=layout)

#plotting graph
plotly.offline.iplot(fig)
#Plot 4
#creating scatter plot (ploty package)
trace1 = go.Scatter(x =results['index'], y = results['s2010'],mode = 'markers',name= 'S can. in 2010',
                    marker=dict(size= 11,opacity=0.8))
trace2 = go.Scatter(x =results['index'], y = results['s2012'],mode = 'markers', name= 'S can. in 2012',
                    marker=dict(size= 11,opacity=0.8))
trace3 = go.Scatter(x =results['index'], y = results['s2014'],mode = 'markers', name= 'S can. in 2014', 
                    marker=dict(size= 11,opacity=0.6))
trace4 = go.Scatter(x =results['index'], y = results['s2016'],mode = 'markers',name= 'S can. in 2016',
                    marker=dict(size= 11,opacity=0.8,color='rgb(204,204,0)'))

#defining plot properties
layout =go.Layout(title='Distribution of Senate candidates by states',legend=dict(orientation="h"),
                  height=650,width=1050,
                  
                  xaxis = dict(title = 'States',showgrid=False,showline=True,zeroline=False,mirror="ticks",
                  ticks="inside",linewidth=2,tickwidth=2,zerolinewidth=2),
                  
                  yaxis = dict(title = 'Number of Senate Candidates',showgrid=True,showline=True,zeroline=False,
                  mirror="ticks",ticks="inside",linewidth=2,tickwidth=2,zerolinewidth=2))

fig = go.Figure(data=[trace1,trace2,trace3,trace4], layout=layout)

#plotting graph
plotly.offline.iplot(fig)
#Importing data for next 2 plots
a = pd.read_csv('../input/congressional-election-expenditures/all_house_senate_2010.csv',usecols=[3,4,5,6,18])
b = pd.read_csv('../input/congressional-election-expenditures/2012.csv',usecols=[3,4,5,6,18])
c = pd.read_csv('../input/congressional-election-expenditures/all_house_senate_2014.csv',usecols=[3,4,5,6,18])
d = pd.read_csv('../input/congressional-election-expenditures/all_house_senate_2016.csv',usecols=[3,4,5,6,18],encoding = 'ISO-8859-1')

#converting all the amounts to float after removing $ and , and (-)negitive sign
a['dis_amo']=a['dis_amo'].str.replace('$', ' ').str.replace(',', '').str.replace('-', '').str.strip().astype(float)
a['dis_amo'] = a['dis_amo'].astype(float)

b['dis_amo']=b['dis_amo'].str.replace('$', ' ').str.replace(',', '').str.replace('-', '').str.strip()
b['dis_amo'] = b['dis_amo'].astype(float)

c['dis_amo']=c['dis_amo'].str.replace('$', ' ').str.replace(',', '').str.replace('-', '').str.strip()
c['dis_amo'] = c['dis_amo'].astype(float)

d['dis_amo']=d['dis_amo'].str.replace('$', ' ').str.replace(',', '').str.replace('-', '').str.strip()
d['dis_amo'] = d['dis_amo'].astype(float)
a.head()
#Plot 4
#Histogram(cufflink package)
cf.set_config_file(offline=True, world_readable=True, theme='pearl')

#defining data frame for Histogram
df = pd.DataFrame({'2010': a['dis_amo'].groupby(a['can_nam']).sum(),
                   '2012': b['dis_amo'].groupby(b['can_nam']).sum(),
                   '2014': c['dis_amo'].groupby(c['can_nam']).sum(),
                   '2016': d['dis_amo'].groupby(d['can_nam']).sum()})
#plotting histogram
df.iplot(kind='histogram', subplots=True, shape=(4, 1),title='Distribution of Expenditure',
         yTitle='Number of Cand.', xTitle='Average Spending')
#finding common candidates in all the years and storing them in new dataframe
common = set.intersection(set(a.can_nam), set(b.can_nam), set(c.can_nam),set(c.can_nam))
w=pd.concat([
    a[a.can_nam.isin(common)],
    b[b.can_nam.isin(common)],
    c[c.can_nam.isin(common)],
    d[d.can_nam.isin(common)],]).sort_values(by='can_nam')

#seperating data by year
c10=w.loc[w['ele_yea'] == 2010]
c12=w.loc[w['ele_yea'] == 2012]
c14=w.loc[w['ele_yea'] == 2014]
c16=w.loc[w['ele_yea'] == 2016]
#Plot 5
#creating box plot(plotly package)
trace1 = go.Box(
    x=c10['dis_amo'].groupby(c10['can_nam']).sum(),name='2010',boxmean='sd'
)
trace2 = go.Box(
    x=c12['dis_amo'].groupby(c12['can_nam']).sum(),name='2012',boxmean='sd'

)
trace3 = go.Box(
    x=c14['dis_amo'].groupby(c14['can_nam']).sum(),name='2014',boxmean='sd'
 
)
trace4 = go.Box(
    x=c16['dis_amo'].groupby(c16['can_nam']).sum(),name='2016',boxmean='sd'

)

layout=go.Layout(title='Average spending by repeating candidates through years',
                  yaxis = dict(title = 'Year'),
                  xaxis = dict(title = 'Average Spending',range=[0,5000000]) )

data = [trace4,trace3,trace2,trace1]

fig = go.Figure(data=data, layout=layout)

#plotting boxplot
plotly.offline.iplot(fig)
#importing data for next plot
a = pd.read_csv('../input/congressional-election-expenditures/all_house_senate_2010.csv',usecols=[3,6,18])

#converting all the amounts to float after removing $ and , and (-)negitive sign
a['dis_amo']=a['dis_amo'].str.replace('$', ' ').str.replace(',', '').str.replace('-', '').str.strip()
a['dis_amo'] = a['dis_amo'].astype(float)

#calculating average spending statewise in a new data frame
pieces= {'count' : a['can_nam'].groupby(a['can_off_sta']).nunique(),
         'amount': a['dis_amo'].groupby(a['can_off_sta']).sum()}

result = pd.concat(pieces, axis=1)
result['average']=result['amount']/result['count']
new=result.reset_index()
new.head()
#Plot 6
#converting to string
for col in new.columns:
    new[col] = new[col].astype(str)
#defining colour scale 
scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\
            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]

#creating text on hover
new['text'] = new['can_off_sta'] + '<br>' +\
    'No. of cand.: '+ new['count']
    
#plotting data    
data = [ dict(
        type='choropleth',
        colorscale = scl,
        autocolorscale = False,
        locations = new['can_off_sta'],
        z = new['average'].astype(float),
        locationmode = 'USA-states',
        text = new['text'],
        marker = dict(
            line = dict (
                color = 'rgb(255,255,255)',
                width = 2
            ) ),
        colorbar = dict(
            title = "Expenditure in USD")
        ) ]

#defining plot properties
layout = dict(
        title = '2010 US Congressional Elections Average Expenditure by State<br>(Hover for breakdown)',
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showlakes = True,
            lakecolor = 'rgb(255, 255, 255)'),
             )
    
fig = dict( data=data, layout=layout )

#plotting boxplot
plotly.offline.iplot(fig)
#importing data for next 4 visualizations
a = pd.read_csv('../input/congressional-election-expenditures/all_house_senate_2010.csv',usecols=[8,4,11,17,14,18,23,15])
b = pd.read_csv('../input/congressional-election-expenditures/2012.csv',usecols=[8,4,11,17,14,18,23,15])
c = pd.read_csv('../input/congressional-election-expenditures/all_house_senate_2014.csv',usecols=[8,4,11,17,14,18,23,15])
d = pd.read_csv('../input/congressional-election-expenditures/all_house_senate_2016.csv',usecols=[8,4,11,17,14,18,23,15],encoding = 'ISO-8859-1')
#converting all the amounts to float after removing $ and , and (-)negitive sign
a['dis_amo']=a['dis_amo'].str.replace('$', ' ').str.replace(',', '').str.replace('-', '').str.strip()
a['dis_amo'] = a['dis_amo'].astype(float)

b['dis_amo']=b['dis_amo'].str.replace('$', ' ').str.replace(',', '').str.replace('-', '').str.strip()
b['dis_amo'] = b['dis_amo'].astype(float)

c['dis_amo']=c['dis_amo'].str.replace('$', ' ').str.replace(',', '').str.replace('-', '').str.strip()
c['dis_amo'] = c['dis_amo'].astype(float)

d['dis_amo']=d['dis_amo'].str.replace('$', ' ').str.replace(',', '').str.replace('-', '').str.strip()
d['dis_amo'] = d['dis_amo'].astype(float)
a.head()
#converting to string
a['dis_dat'] = a['dis_dat'].astype('str')
b['dis_dat'] = b['dis_dat'].astype('str')
c['dis_dat'] = c['dis_dat'].astype('str')
d['dis_dat'] = d['dis_dat'].astype('str')
#parsing date and time for all data frames
##putting in new cell because every cell takes a lot of time to run 
###Be patient :-)

a['dis_dat'] = a['dis_dat'].apply(lambda x : pd.to_datetime(x , format='%Y-%m-%d', errors='coerce'))
b['dis_dat'] = b['dis_dat'].apply(lambda x : pd.to_datetime(x , format='%Y-%m-%d', errors='coerce'))
c['dis_dat'] = c['dis_dat'].apply(lambda x : pd.to_datetime(x , format='%Y-%m-%d', errors='coerce'))
d['dis_dat'] = d['dis_dat'].apply(lambda x : pd.to_datetime(x , format='%Y-%m-%d', errors='coerce'))
#replacing date with month
a['dis_dat'] = pd.DatetimeIndex(a['dis_dat']).month
b['dis_dat'] = pd.DatetimeIndex(b['dis_dat']).month
c['dis_dat'] = pd.DatetimeIndex(c['dis_dat']).month
d['dis_dat'] = pd.DatetimeIndex(d['dis_dat']).month
a.head()
#grouping amount by month
pieces = {'2010': a['dis_amo'].groupby(a['dis_dat']).mean(),
          '2012': b['dis_amo'].groupby(b['dis_dat']).mean(), 
          '2014': c['dis_amo'].groupby(c['dis_dat']).mean(), 
          '2016': d['dis_amo'].groupby(d['dis_dat']).mean()}

#merging results in a new dataframe
result = pd.concat(pieces, axis=1)
results=result.reset_index()
results
#Plot 7
#defining dim for each plot(total plots 4)
a4_dims = (20, 8)

fig, (ax1, ax2,ax3,ax4) = plt.subplots(ncols=4,figsize=a4_dims, sharey=True)

month=["Jan","Feb","Mar","Apr","May","June","July","Aug","Sep","Oct","Nov","Dec"]

#first bar chart
pal = sns.color_palette("Blues_d", len(results['2010']))
rank = results['2010'].argsort().argsort()
sns.barplot(x=month, y=results['2010'], ax=ax1, palette=np.array(pal[::-1])[rank])
ax1.set(xlabel='Months', ylabel='Average Spending',title="Average Disbursment-2010 (by months)")

#second bar chart
pal = sns.color_palette("Blues_d", len(results['2012']))
rank = results['2012'].argsort().argsort()
sns.barplot(x=month, y=results['2012'], ax=ax2, palette=np.array(pal[::-1])[rank])
ax2.set(xlabel='Months', ylabel='Average Spending',title="Average Disbursment-2012 (by months)")

#third bar chart
pal = sns.color_palette("Blues_d", len(results['2014']))
rank = results['2014'].argsort().argsort()
sns.barplot(x=month, y=results['2014'], ax=ax3, palette=np.array(pal[::-1])[rank])
ax3.set(xlabel='Months', ylabel='Average Spending',title="Average Disbursment-2014 (by months)")

#fourth bar chart
pal = sns.color_palette("Blues_d", len(results['2016']))
rank = results['2016'].argsort().argsort()
sns.barplot(x=month, y=results['2016'], ax=ax4, palette=np.array(pal[::-1])[rank])
ax4.set(xlabel='Months', ylabel='Average Spending',title="Average Disbursment-2016 (by months)")

#plotting graph
sns.plt.show(fig)

newa = a[["cat_des", "dis_amo","dis_dat"]].copy()
result=newa.groupby(["cat_des","dis_dat"]).sum()
results=result.reset_index()
results.head()
#plot 8
#2d-numpy from data frame for scatter plot input
a1=results['dis_amo'].iloc[0:].values.reshape(12,12)

#making a numpy for month labels on x
month=["Jan","Feb","Mar","Apr","May","June","July","Aug","Sep","Oct","Nov","Dec"]
#plotting scatter plot(seaborn)
sns.set()
ax = sns.heatmap(a1,linewidths=.5, xticklabels=month, yticklabels=results['cat_des'].unique())
ax.set(xlabel='Months', ylabel='Category',title="Type of Disbursment per month(2010)")

#copying data to new data frame for calculations
newa = a.filter(['cat_des','lin_num','dis_amo'], axis=1)

#creating pivot table from new dataframe
result = pd.pivot_table(newa, values='dis_amo', index=['cat_des'],columns=['lin_num'], aggfunc=np.sum)

#filling with zeros for na values
result.fillna(0, inplace=True)

#creating series from pivot table
s = result.stack()

#creating data frame from series 
x=s.to_frame()
y=x.reset_index()
y.info()

#2d-numpy from data frame for scatter plot input
a1=y[0].iloc[0:].values.reshape(12,8)

#plotting scatter plot(seaborn)
sns.set()
ax = sns.heatmap(a1,linewidths=.5, xticklabels=newa['lin_num'].unique(), yticklabels=y['cat_des'].unique(),
                 cmap="Oranges") 

ax.set(xlabel='FEC Line Number', ylabel='Category',title="FEC Line number corresponding to expense type")
#appendind states to cities and storing it in new column as add
a['add']=a['rec_cit']+","+a['rec_sta']
b['add']=b['rec_cit']+","+b['rec_sta']
c['add']=c['rec_cit']+","+c['rec_sta']
d['add']=d['rec_cit']+","+d['rec_sta']
a.head()
#sum of dis. amount and count of recepients by address(city)-2010
pieces={'amount':a['dis_amo'].groupby(a['add']).sum(),
        'rcount':a['rec_nam'].groupby(a['add']).nunique()}
result=pd.concat(pieces,axis=1)
p=result.sort_values(by='amount',ascending=False).reset_index()[:10]
p['year']=2010
#sum of dis. amount and count of recepients by address(city)-2012
pieces={'amount':b['dis_amo'].groupby(b['add']).sum(),
        'rcount':b['rec_nam'].groupby(b['add']).nunique()}
result=pd.concat(pieces,axis=1)
q=result.sort_values(by='amount',ascending=False).reset_index()[:10]
q['year']=2012
#sum of dis. amount and count of recepients by address(city)-2014
pieces={'amount':c['dis_amo'].groupby(c['add']).sum(),
        'rcount':c['rec_nam'].groupby(c['add']).nunique()}
result=pd.concat(pieces,axis=1)
r=result.sort_values(by='amount',ascending=False).reset_index()[:10]
r['year']=2014
#sum of dis. amount and count of recepients by address(city)-2016
pieces={'amount':d['dis_amo'].groupby(d['add']).sum(),
        'rcount':d['rec_nam'].groupby(d['add']).nunique()}
result=pd.concat(pieces,axis=1)
s=result.sort_values(by='amount',ascending=False).reset_index()[:10]
s['year']=2016
#concatenaing data frames
result = pd.concat([p,q,r,s])
results=result.reset_index()
results.tail()
######################################## Run this only if you have google API key #############################################
########################################### or else use the csv file attached #################################################

#getting lat long from cities
#os.environ["ENTER_YOUR_API_KEY"] = "api_key_from_google_cloud_platform"
'''
countCity = 0
locs = []
for addr in results['add']:
    countCity += 1
    if(countCity % 10 == 0):
        time.sleep(3)
    locs.append(Geocoder.geocode(addr))

    #storing it in data frame
    g= pd.DataFrame(
    [ (addr.latitude, addr.longitude) for addr in locs ],
    columns=['latitude', 'longitude'])
'''

#reading latitude/longitude from csv file
g=pd.read_csv('../input/lat-long-congressional-disbursements/latlng.csv')
#mergind cities and lat/long
final=pd.concat([results,g], axis=1)
final.head()
#text for hover
final['text'] = final['add'] + '<br>Revenue: ' + (final['amount']/1e6).astype(str)+' million'+ '<br>Total Recepients: ' + \
                (final['rcount']).astype(str)

#setting limits for color according to year
limits = [(0,9),(10,19),(20,29),(30,39)]

#color scale
colors = ["rgb(0,116,217)","rgb(255,65,54)","rgb(133,20,75)","rgb(173, 244, 66)"]
cities = []
scale = 700000

#Mapping points on map 
for i in range(len(limits)):
    lim = limits[i]
    df_sub = final[lim[0]:lim[1]]
    city = dict(
        type = 'scattergeo',
        locationmode = 'USA-states',
        lon = df_sub['longitude'],
        lat = df_sub['latitude'],
        text = df_sub['text'],
        marker = dict(
            
            size = df_sub['amount']/scale,
            color = colors[i],
            line = dict(width=0.5, color='rgb(40,40,40)'),
            sizemode = 'area'
        ),
         name=df_sub['year'].iloc[0]) 
    cities.append(city)

#plot properties
layout = dict(
        title = 'Top 10 cities generating Highest Revenues during Elections<br>(Click legend in order 2010 -> 2016 to see changes)',
        showlegend = True,
        geo = dict(
            scope='usa',
            projection=dict( type='albers usa' ),
            showland = True,
            landcolor = 'rgb(217, 217, 217)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        ),
    )

fig = dict( data=cities, layout=layout)

#plotting map
plotly.offline.iplot( fig, validate=False)

