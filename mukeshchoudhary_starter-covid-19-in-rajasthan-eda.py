import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.graph_objects as go
cf.go_offline()
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['District'], y= data['Cumulative_Positive'], name='Positive_Cases',
                         line=dict(color='red', width=2)))
fig.add_trace(go.Scatter(x=data['District'], y=data['Recoverd'], name='Recovered',
                         line=dict(color='blue', width=2)))
fig.add_trace(go.Scatter(x=data['District'], y=data['Discharged'], name='Discharged',
                         line=dict(color='green', width=2)))

fig.update_layout(
    title='Corona Virus Trend in Rajasthan',
     yaxis=dict(
        title='Number of Cases')
    )

fig.show()

fig = go.Figure()
fig.add_trace(go.Scatter(x=data['District'], y= data['Cumulative_Positive'], name='Positive_Cases',
                         line=dict(color='red', width=2)))
data.sort_values('Cumulative_Positive',ascending=False)[:10].iplot(kind='bar',
                                                                               x='District',
                                                                               color = ['blue','green','red'],
                                                                               title='Top 10 Districts with Total Confirmed Cases',
                                                                               xTitle='Districts',
                                                                               yTitle = 'Cases Count')
data.sort_values('Recoverd',ascending=False)[:10].iplot(kind='bar',
                                                                               x='District',
                                                                               color = ['blue','green','red'],
                                                                               title='Top 10 Districts with Total Recovered Cases',
                                                                               xTitle='Districts',
                                                                               yTitle = 'Cases Count')
corr = data[['Cumulative_Positive','Recoverd','Discharged']].corr()
mask = np.triu(np.ones_like(corr,dtype = bool))

plt.figure(dpi=100)
plt.title('Correlation Analysis')
sns.heatmap(corr,mask=mask,annot=True,lw=1,linecolor='white',cmap='Reds')
plt.xticks(rotation=0)
plt.yticks(rotation = 0)
plt.show()
