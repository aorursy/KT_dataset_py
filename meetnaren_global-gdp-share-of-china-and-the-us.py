import pandas as pd
from requests import get
from bs4 import BeautifulSoup

d=pd.read_csv('../input/GDPsharedata (3).csv')

alldata=pd.DataFrame(columns=['year','country','gdpshare'])
!ls ../input/
hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
         'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
         'Referer': 'https://cssspritegenerator.com',
         'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
         'Accept-Encoding': 'none',
         'Accept-Language': 'en-US,en;q=0.8',
         'Connection': 'keep-alive'}
for year in range(1980,2020):
    data_url='http://www.economywatch.com/economic-statistics/economic-indicators/GDP_Share_of_World_Total_PPP/'+str(year)+'/'
   
    page=get(data_url, headers=hdr)
    soup = BeautifulSoup(page.content, 'html.parser')
    table=soup.find('table',attrs={'id':'tbl_sort'})
    tr=table.tbody.find_all('tr')
  
    for row in tr: 
        data=row.find_all('td')
        country=data[1].string
        gdpshare=data[3].string.split()[0]
        alldata.loc[len(alldata)]=[year,country,gdpshare]
alldata=pd.read_csv("../input/GDPsharedata (3).csv")
alldata.head()
ChinaUSdata=alldata[(alldata['country']=='China') | (alldata['country']=='United States')]
Chinadata=alldata[(alldata['country']=='China')]
USdata=alldata[(alldata['country']=='United States')]
ChinaUSdata.head()
import plotly.offline as offline
import plotly.graph_objs as go

offline.init_notebook_mode()

China = go.Scatter(
    x=Chinadata['year'], 
    y=Chinadata['gdpshare'],
    mode='lines',
    line=dict(color='rgba(0,0,255,1)'),
    name='China'
)
US = go.Scatter(
    x=USdata['year'], 
    y=USdata['gdpshare'],
    mode='lines',
    line=dict(color='rgba(255,0,0,1)'),
    name='US'
)

Chinaendpoints = go.Scatter(
    x=[Chinadata['year'].iloc[0], Chinadata['year'].iloc[-1]],
    y=[Chinadata['gdpshare'].iloc[0], Chinadata['gdpshare'].iloc[-1]],
    mode='markers',
    marker=dict(color='rgba(0,0,255,1)')
)

USendpoints = go.Scatter(
    x=[USdata['year'].iloc[0], USdata['year'].iloc[-1]],
    y=[USdata['gdpshare'].iloc[0], USdata['gdpshare'].iloc[-1]],
    mode='markers',
    marker=dict(color='rgba(255,0,0,1)')
)
plotdata=[China, US, Chinaendpoints, USendpoints]

layout = go.Layout(
    xaxis=dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        tickwidth=2,
        ticklen=5,
        title='Year'
    ),
    yaxis=dict(
        showgrid=False,
        zeroline=False,
        showline=True,
        showticklabels=True,
        title='% of global GDP'
    ),
    showlegend=False,
    annotations=[
        dict(
          x=Chinadata['year'].iloc[-1],
          y=Chinadata['gdpshare'].iloc[-1],
          text='China',
            showarrow=False,
            xanchor='center',
            yanchor='top'
        ),
        dict(
          x=USdata['year'].iloc[-1],
          y=USdata['gdpshare'].iloc[-1],
          text='US',
            showarrow=False,
            xanchor='center',
            yanchor='top'
        ),
    ]
)
offline.init_notebook_mode(connected=False)

fig = go.Figure(data=plotdata, layout=layout)

offline.iplot(fig)
