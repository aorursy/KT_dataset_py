import pandas as pd
import plotly.express as px

data = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header = 0)
data = data.groupby(['ObservationDate', 'Country/Region']).sum().reset_index()

#creating anmation that represents deaths due to Corona-Virus worldwide
fig = px.choropleth(data, locations = 'Country/Region', locationmode = 'country names', color = 'Deaths', hover_name = 'Country/Region', animation_frame = 'ObservationDate', color_continuous_scale = 'Rainbow', range_color = (0.1, 100000.) )
fig.update_layout(title_text = 'Dolgormaa_Banzragch', title_x = 0.5)
fig.show()
#fig.write_html('Dolgormaa_banzragch_20M51866.html')
import pandas as pd
import plotly.express as px
df = pd.read_csv('../input/2015april-banza/query (2).csv', header = 0)

df.index = pd.to_datetime(df['time'])
df['time'] = df.index.strftime('%Y-%m-%d %H:00:00')
fig1 = px.scatter_geo(df,lat = 'latitude', lon = 'longitude', color = 'mag', animation_frame = 'time', color_continuous_scale = 'Rainbow', range_color =  (5.,7.))
fig1.update_layout(title_text = 'Dolgormaa_Banzragch', title_x = 0.5)
fig1.show()
#fig1.write_html('2015_April_#20M51866.html')