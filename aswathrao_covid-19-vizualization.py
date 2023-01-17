#!pip install google.colab
from datetime import datetime
from collections import defaultdict
import requests
import operator
import pandas as pd
import matplotlib.pyplot as plt
#import plotly.graph_objects as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 
## fetching the raw data
r = requests.get('https://api.covid19india.org/raw_data.json')
jsonData = r.json()


listJsonData = jsonData.get('raw_data')
Coviddf = pd.DataFrame.from_dict(listJsonData, orient='columns')
#print(Coviddf)
RawCoviddf= Coviddf
RawCoviddf['dateannounced'] = pd.to_datetime(RawCoviddf['dateannounced'],format ='%d/%m/%Y')
RawCoviddf.dropna(subset=['detectedstate'],inplace = True)

# recovered
url1 = 'https://api.covid19india.org/states_daily_csv/recovered.csv'
recDf = pd.read_csv(url1)
recDf = recDf.iloc[:,0:recDf.shape[1]-1]
recDf.fillna(0,inplace = True)
# deceased
url2 = 'https://api.covid19india.org/states_daily_csv/deceased.csv'
decDf =pd.read_csv(url2)
decDf = decDf.iloc[:,0:decDf.shape[1]-1]
decDf.fillna(0,inplace = True)
# confirmed
url3 = 'http://api.covid19india.org/states_daily_csv/confirmed.csv'
conDf = pd.read_csv(url3)
conDf = conDf.iloc[:,0:conDf.shape[1]-1]
conDf.fillna(0,inplace = True)
# state
stateDict  = {'AP': 'Andhra Pradesh','AR': 'Arunachal Pradesh','AS': 'Assam','BR': 'Bihar','CT': 'Chhattisgarh','GA': 'Goa',
              'GJ': 'Gujarat','HR': 'Haryana','HP': 'Himachal Pradesh','JH': 'Jharkhand','KA': 'Karnataka','KL': 'Kerala',
              'MP': 'Madhya Pradesh','MH': 'Maharashtra','MN': 'Manipur','ML': 'Meghalaya','MZ': 'Mizoram','NL': 'Nagaland',
              'OR': 'Odisha','PB': 'Punjab','RJ': 'Rajasthan','SK': 'Sikkim','TN': 'Tamil Nadu','TG': 'Telangana','TR': 'Tripura',
              'UT': 'Uttarakhand','UP': 'Uttar Pradesh','WB': 'West Bengal','AN': 'Andaman and Nicobar Islands','CH': 'Chandigarh',
              'DN': 'Dadra and Nagar Haveli','DD': 'Daman and Diu','DL': 'Delhi','JK': 'Jammu and Kashmir','LA': 'Ladakh','LD': 'Lakshadweep',
              'PY': 'Puducherry'}
combinedDf = pd.DataFrame(columns= ['Date','State','Confirmed','Recovered','Dead'])
index = 0
for i in range(len(recDf)):
    for j in range(2,len(recDf.iloc[i])):
        if conDf.iloc[i,j]==0 and recDf.iloc[i,j]==0 and decDf.iloc[i,j]==0:
            continue
        record = [recDf['date'][i],stateDict.get(recDf.columns[j]),int(conDf.iloc[i,j]),int(recDf.iloc[i,j]),int(decDf.iloc[i,j])]
        combinedDf.loc[index]=record
        index=index+1
index = 0
for i in range(len(recDf)):
    for j in range(2,len(recDf.iloc[i])):
        if conDf.iloc[i,j]==0 and recDf.iloc[i,j]==0 and decDf.iloc[i,j]==0:
            continue
        record = [recDf['date'][i],stateDict.get(recDf.columns[j]),int(conDf.iloc[i,j]),int(recDf.iloc[i,j]),int(decDf.iloc[i,j])]
        combinedDf.loc[index]=record
        index=index+1

obj = datetime.strptime(combinedDf['Date'][len(combinedDf)-1],'%d-%b-%y')

# getting the data till Yesterday
RawCoviddf = RawCoviddf[RawCoviddf['dateannounced']<=datetime.strftime(obj,'%Y-%m-%d')]
combinedDf['Date'] = pd.to_datetime(combinedDf['Date'],format ='%d-%b-%y')
stateCount = defaultdict(list)

# state Wise summing up
for i in range(len(combinedDf)):

    #Active case calculation
    value = combinedDf['Confirmed'][i] -(combinedDf['Recovered'][i]+combinedDf['Dead'][i])
    
    if combinedDf['State'][i] not in stateCount:
        stateCount[combinedDf['State'][i]].append(combinedDf['Confirmed'][i])
        stateCount[combinedDf['State'][i]].append(combinedDf['Recovered'][i])
        stateCount[combinedDf['State'][i]].append(combinedDf['Dead'][i])
        stateCount[combinedDf['State'][i]].append(value)
    else:
        stateCount[combinedDf['State'][i]][0]+=combinedDf['Confirmed'][i]
        stateCount[combinedDf['State'][i]][1]+=combinedDf['Recovered'][i]
        stateCount[combinedDf['State'][i]][2]+=combinedDf['Dead'][i]
        stateCount[combinedDf['State'][i]][3]+=value


# sorting in reverse order to get the state with maximum cases
stateCount = dict(sorted(stateCount.items(), key = lambda x :x[1][0], reverse=True ))
stateCount
Tabulation = pd.DataFrame.from_dict(stateCount,orient='index',columns=list(combinedDf.columns[2:])+['Active'])
Tabulation
#calculation Part
value = list(Tabulation.sum(axis=0,skipna=True))

# display using plotly pie chart
colors = ['red','green','grey']
fig = go.Figure(data=go.Pie(values = value[:-1], labels = Tabulation.columns.tolist(),textinfo='percent+label',marker=dict(colors=colors)))
fig.update_layout(title_text='Affect of COVID 19 in India')

fig.show()
# calculation Part
df = combinedDf.groupby('Date').sum().groupby('Date').cumsum()
df['Confirmed'] = df['Confirmed'] - (df['Recovered']+df['Dead'])
df.columns =['Active','Recovered','Dead']

#display
fig = go.Figure(data=[
    go.Bar(name='Dead', x=df.index.tolist(), y=df['Dead'],marker_color = 'grey'),
    go.Bar(name='Recovered', x=df.index.tolist(), y=df['Recovered'],marker_color = 'green'),
    go.Bar(name='Active', x=df.index.tolist(), y=df['Active'],marker_color = 'red')
])

fig.update_layout(barmode='stack')
fig.update_layout(title_text='Daily cases of Covid-19 in India',yaxis=dict(title='Number of cases recorded' ),xaxis=dict(title='Date' ))

fig.show()
#calculation Part
df = combinedDf.groupby('Date').sum().groupby('Date').cumsum()
cumDate= df.cumsum()
cumDate

#display
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=cumDate.index.tolist(), y=cumDate['Dead'],
    hoverinfo='x+y',
    mode='lines+markers',
     line_color='grey',
    stackgroup='one',
    name = "Dead"
))

fig.add_trace(go.Scatter(x=cumDate.index.tolist(), y=cumDate['Recovered'],
    hoverinfo='x+y',
    mode='lines+markers',
    line_color='green',
    stackgroup='one',
    name = "Recovered"
))
fig.add_trace(go.Scatter( x=cumDate.index.tolist(), y=cumDate['Confirmed'],
    hoverinfo='x+y',
    mode='lines+markers',
    line_color='red',
    stackgroup='one',
    name ="Confirmed"
))
fig.update_layout(title_text='Cummulative Trends of Covid-19 India',yaxis=dict(title='Number of cases' ),xaxis=dict(title='Date'))
fig.show()
#calculation Part
growthRate=dict()
for i in range(len(combinedDf)):
    
    if combinedDf['State'][i] not in growthRate:
        growthRate[combinedDf['State'][i]]=0

for i in growthRate.items():
    l = []
    for j in range(len(combinedDf)):
        if i[0] == combinedDf['State'][j]:
            l.append(combinedDf['Confirmed'][j])
    res = sum(list(map(operator.sub, l[1:], l[:-1])))
    if res >0:
        growthRate[i[0]]+=res
growthRate = dict(sorted(growthRate.items(), key = lambda x :x[1], reverse=True ))
label = list(growthRate.keys())
value = list(growthRate.values())
total = sum(value)
value = [ round(((value[i]/total)*100),2)for i in range(len(value))]

#display
fig = go.Figure(go.Bar(
            x=value,
            y=label,
            orientation='h'))
          
fig.update_layout(title_text='State Wise Growth Rate of Covid-19 India',yaxis=dict(title='States',autorange="reversed"),xaxis=dict(title='percentage'))
fig.show()

#calculation Part
tcdf = combinedDf.groupby(['Date','State']).sum().groupby(['State']).cumsum()
tcdf = tcdf.reset_index()

#display
fig = go.Figure()
for i in Tabulation.index.tolist():
    l=[]
    for j in range(len(tcdf)):
        if tcdf['State'][j]==i:
            l.append(tcdf['Confirmed'][j])
    fig.add_trace(go.Scatter(x=cumDate.index.tolist(), y=l,
        mode='lines',name=i))
fig.update_layout(title_text='State Wise confirmed cases of Covid-19 in India',yaxis=dict(title='Number of cases counted' ),xaxis=dict(title='Date' ))
fig.show()
#calculation Part & display
fig = go.Figure()
for i in Tabulation.index.tolist():
    l=[]
    for j in range(len(tcdf)):
        if tcdf['State'][j]==i:
            l.append(tcdf['Recovered'][j])
    fig.add_trace(go.Scatter(x=cumDate.index.tolist(), y=l,
                    mode='lines',
                    name=i))
fig.update_layout(title_text='State Wise recovered cases of Covid-19 India',yaxis=dict(title='Number of cases' ),xaxis=dict(title='Date' ))
fig.show()
fig = go.Figure()
for i in Tabulation.index.tolist():
    l=[]
    for j in range(len(tcdf)):
        if tcdf['State'][j]==i:
            l.append(tcdf['Dead'][j])
    fig.add_trace(go.Scatter(x=cumDate.index.tolist(), y=l,
                    mode='lines',
                    name=i))
fig.update_layout(title_text='State Wise dead cases of Covid-19 India',yaxis=dict(title='Number of cases' ),xaxis=dict(title='Date' ))
fig.show()
#calculation Part
import numpy as np
riDf = combinedDf.groupby('Date').sum()
recoveryRate = round((sum(combinedDf['Recovered'])+sum(combinedDf['Dead']))/sum(combinedDf['Confirmed']),2)

grratelist =[]
for i in range(len(riDf)-1):
    grratelist.append(round(riDf['Confirmed'][i+1]/riDf['Confirmed'][i],2))

rrratelist= []
for i in range(len(cumDate)):
    rrratelist.append(round(((cumDate['Recovered'][i]+cumDate['Dead'][i])/cumDate['Confirmed'][i]),2))

#display
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=cumDate.index.tolist()[:-1], y=rrratelist[:-1],
    hoverinfo='x+y',
    mode='lines+markers',
     line_color='green',
    stackgroup='one',
    name = "recovery"
))
fig.add_trace(go.Scatter(
    x=riDf.index.tolist()[:-1], y=grratelist,
    hoverinfo='x+y',
    mode='lines+markers',
     line_color='red',
    stackgroup='one',
    name = "growth"
))

print("Average Rate of growth : ",round(np.mean(rrratelist),2))
fig.update_layout(title_text='Daily Growth Rate and Recovery Rate of Covid-19 in India',yaxis=dict(title='Rate' ),xaxis=dict(title='Date' ))
fig.show()