import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
#plotly.tools.set_credentials_file(username='username', api_key='api_key')
#To run this code in your own notebook you have to change 'username' and 'api_key' by your own values after having an account on https://plot.ly/
data =  pd.read_csv('../input/BCT_statistque.csv',sep =",")
data.head()
# Create traces
variables = data.columns.values
p = []
for i in range(1,len(variables)):
    a = go.Scatter(
        x = data.Date,
        y = data[variables[i]],
        mode = 'lines+markers',
        name = variables[i])
    p.append(a)

#py.iplot(p, filename='line-mode')



