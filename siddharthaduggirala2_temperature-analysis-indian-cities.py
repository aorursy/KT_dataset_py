import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')


global_data = pd.read_csv('../input/GlobalLandTemperaturesByMajorCity.csv')

indian_data = global_data[global_data.Country.isin(['India'])]
cities = np.unique(indian_data['City'])

mean_cities_temperature_bar, cities_bar = (list(x) for x in zip(*sorted(zip(mean_cities_temperature,cities), reverse=True)))

sns.set(font_scale=0.9)
f, ax = plt.subplots(figsize=(2, 6))
colors_cw = sns.color_palette('coolwarm', len(cities))
sns.barplot(mean_cities_temperature_bar, cities_bar, palette = colors_cw[::-1])
Text = ax.set(xlabel='Average temperature', title='Average land temperature in cities')

# looks like Madras, Hyderabad and Bombay are top 3 cities with high temperatures.
years = np.unique(indian_data['dt'].apply(lambda x: x[:4]))

mean_indian_temp= []
mean_indian_temp_un = []


for year in years:
   mean_indian_temp.append(indian_data[indian_data['dt'].apply(
       lambda x: x[:4]) == year]['AverageTemperature'].mean())
   mean_indian_temp_un.append(indian_data[indian_data['dt'].apply(
       lambda x: x[:4]) == year]['AverageTemperatureUncertainty'].mean())


line1 = go.Scatter(
    x = years,
    y = np.array(mean_indian_temp) + np.array(mean_indian_temp_un),
    fill = None,
    mode = 'lines',
    name = 'Uncertainty top',
    line=dict(
        color='rgb(0, 255, 255)',)
    )

line2 = go.Scatter(
    x = years,
    y = np.array(mean_indian_temp) - np.array(mean_indian_temp_un),
    mode = 'lines',
    name = 'Uncertainty bot',
    line=dict(
        color='rgb(0, 255, 255)',)
)

line3 = go.Scatter(
    x = years,
    y = np.array(mean_indian_temp),
    line=dict(
        color='rgb(199, 121, 093)',)
)

line = [line1,line2,line3]

layout = go.Layout(xaxis=dict(title='year'),
    yaxis=dict(title='Average Temperature, °C'),
    title='Average land temperature in Indian Cities',
    showlegend = False)

fig = go.Figure(data=line, layout=layout)
py.iplot(fig)

#Average Temperature across India
india_temp=indian_data[['dt','AverageTemperature']]
india_temp['dt']=pd.to_datetime(india_temp.dt).dt.strftime('%d/%m/%Y')
india_temp['dt']=india_temp['dt'].apply(lambda x:x[6:])
india_temp=india_temp.groupby(['dt'])['AverageTemperature'].mean().reset_index()
trace=go.Scatter(
    x=india_temp['dt'],
    y=india_temp['AverageTemperature'],
    mode='lines',
    )
data=[trace]

py.iplot(data, filename='line-mode')
#Temperature clearly shows upward trend in India