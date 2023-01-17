import pandas as pd
import plotly.graph_objs as go 
import seaborn as sns
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
%matplotlib inline
df = pd.read_csv('../input/web-visitor-interests/visitor-interests.csv')
df.head(5)
df_code = pd.read_csv('../input/countrycode-from-ibancom/CountryCode')
df_code.head()
dx=df_code[['Country','Alpha-2 code','Alpha-3 code']]
dx.columns = ['Country full name','Country','Alpha-3 code']
df_all = pd.merge(df,dx,on='Country').dropna(axis=0)
df_all.head()
df_IP = df_all.groupby('Alpha-3 code',as_index=False).count()
df_IP.head()
data = dict(
        type = 'choropleth',
        locations = df_IP['Alpha-3 code'],
        z = df_IP['IP'],
        text = df_IP['IP'],
        colorbar = {'title' : 'Number of visitors'},
    colorscale='Oranges'
    ,
      ) 
layout = dict(
    title = 'Number of Visitors',
    geo = dict(
        showframe = False,
        projection = {'type':'equirectangular'}
    )
)
choromap = go.Figure(data = [data],layout = layout)
iplot(choromap)
df_all.groupby('Country full name')['IP'].count().sort_values(ascending=False).head(5)
df_bylang = df_all.groupby(by='Languages',as_index=False)['IP'].count()
df_bylang.sort_values('IP',ascending=False).head(5)
df_byint = df_all.groupby(by='Interests',as_index=False)['IP'].count()
df_byint.sort_values('IP',ascending=False).head(5)
#English
df_all[df_all['Languages']=='english'].groupby('Country full name').count().sort_values(by='IP',ascending=False).head(5)
#Russian
df_all[df_all['Languages']=='russian'].groupby('Country full name').count().sort_values(by='IP',ascending=False).head(5)
#Chinese
df_all[df_all['Languages']=='chinese'].groupby('Country full name').count().sort_values(by='IP',ascending=False).head(5)
#French
df_all[df_all['Languages']=='french'].groupby('Country full name').count().sort_values(by='IP',ascending=False).head(5)