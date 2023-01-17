import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import cufflinks as cf
cf.set_config_file(offline=True, world_readable=True, theme='ggplot')

import matplotlib.pyplot as plt
import datetime as dt
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#dateparse = lambda d : dt.datetime.strptime(parse(d), '%d/%m/%Y')
data  = pd.read_csv("../input/startup_funding.csv",thousands=',',usecols=range(1,10))
data.head()
for col in data.columns[:7]:
    data[col].fillna(method='ffill',inplace =True)
    data[col] = data[col].map(lambda x: str(x).capitalize().strip())
data['CityLocation'] = data['CityLocation'].map(lambda x: x.split('/')[0].strip())    
data['Date'] = pd.to_datetime(data['Date'],errors='coerce')
data['Year'] = data['Date'].dt.strftime('%Y')
data.info()
my_colors = ['green','orange','yellowgreen','red','blue']  
xs = ['IndustryVertical', 'SubVertical','CityLocation', 'InvestorsName', 'InvestmentType']
for col, c, bc in zip(xs,my_colors,my_colors[::-1]):
    ttl = f'Top 20 {col} wise distribution'
    data[col].value_counts()[:20][::-1].iplot(kind = 'barh',yTitle="Frequency",xTitle=col,title=ttl,color =bc)
    
data['AmountInUSD'].fillna(0,inplace=True)
data['AmountInUSD'] = data['AmountInUSD']/1000000

xs = ['StartupName', 'IndustryVertical', 'SubVertical',
       'CityLocation', 'InvestorsName', 'InvestmentType']
for col, c in zip(xs,my_colors[::-1]+['navy']):
    ttl =f'Top 20 {col} funds (in Million)'
    data.groupby(col).sum().sort_values(by=["AmountInUSD"], ascending=False)[:20][::-1].iplot(kind='barh',title=ttl,xTitle='Amount in USD',yTitle=col,color=c)
    
#d1 = d.sort_values(by=["AmountInUSD","CityLocation"], ascending=[False, False])[:10]
#d = data.groupby('IndustryVertical')['SubVertical'].value_counts()
data.groupby('Year').sum().iplot(kind='bar',title='Year wise funds (in Million)',xTitle='Year',yTitle='Amount in USD (Million)')
years = ['2015','2016','2017']
impcities = ['Bangalore', 'Gurgaon', 'Mumbai', 'New delhi', 'Pune', 'Hyderabad',
       'Noida', 'Chennai', 'Ahmedabad', 'Kolkata','Jaipur','Chandigarh']
l = []
for y,c in zip(years,['r','g','blue']):
    d = data[data['Year'] == y].groupby('CityLocation').sum().sort_values(by=["AmountInUSD"], ascending=False)['AmountInUSD']
    impval = d[impcities].values
    l.append(impval)
    print(f'In year {y}, {len(d.index)} cities received fund.')
%matplotlib agg
df = pd.DataFrame(l,index=years,columns=impcities)
df.iplot(kind='bar',title='12 imp. cities vs fund division per year',xTitle='Year',yTitle='Amount in USD (Million)')
paytm = data[data['StartupName']=='Paytm']
paytm
undisclosed = data[data['InvestorsName']=='Undisclosed investors']
undisclosed.head()
xs = ['StartupName', 'IndustryVertical', 'SubVertical','CityLocation', 'InvestmentType']
for col in xs:
    ttl =f'Top Indian {col}s having Undisclosed Investors'
    undisclosed.groupby(col).sum()['AmountInUSD'].sort_values(ascending=False)[:20].iplot(kind='bar',title=ttl,xTitle=col,yTitle='Amount in USD (Million)',color='navy')