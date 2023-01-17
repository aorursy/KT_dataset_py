from IPython.display import HTML

HTML('<center><iframe width="1077" height="721" src="https://www.youtube.com/embed/Atrx1P2EkiQ" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></center>')
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.graph_objects as go

import plotly.express as px



import os

for dirname, _, filenames in os.walk('/kaggle/input/who-immunization-coverage'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/who-immunization-coverage/BCG.csv', skipinitialspace=True)

df
# not necessary

country_names = {'Russian Federation':'Russia', "Democratic People's Republic of Korea":"Korea, Democratic People's Republic of", 

                 'Republic of North Macedonia':'Macedonia, the former Yugoslav Republic of', 'Bolivia (Plurinational State of)': 'Bolivia',

                'Cabo Verde':'Cape Verde', 'Congo' : 'Congo (Brazzaville)', 'Czechia' : 'Czech Republic', 'Democratic Republic of the Congo' : 'Congo (Kinshasa)',

                'Eswatini': 'Swaziland', 'Iran (Islamic Republic of)': 'Iran', 'Libya' : 'Libyan Arab Jamahiriya', 'Micronesia (Federated States of)': 'Micronesia, Federated States of',

                'United Republic of Tanzania' : 'Tanzania', 'Venezuela (Bolivarian Republic of)' : 'Venezuela', 'Viet Nam':'Vietnam'}

df['Country'].replace(country_names, inplace = True)
df=df.melt(id_vars=['Country'], var_name='Year', value_name='Percent')

df = df.sort_values(by=['Country', 'Year'], ascending=[True, False])

df
df.isnull().sum(axis = 0)
df = df.dropna()

df
px.choropleth(df, locations=df['Country'], locationmode='country names', color = df['Percent'], hover_name=df['Country'], animation_frame=df['Year'],

              color_continuous_scale=px.colors.sequential.RdBu, projection='natural earth')
lastyear = df.groupby('Country').head(1)

lastyear
fig = go.Figure(data=go.Choropleth(

    locations = lastyear['Country'],

    z = lastyear['Percent'],

    text = lastyear['Year'] + ', ' + lastyear['Country'],

    colorscale = 'RdBu',

    marker_line_color='darkgray',

    marker_line_width=0.5,

    colorbar_title = 'BCG, %',

    locationmode='country names',

))



fig.update_layout(

    title_text='Latest BCG reported woldwide'

)
fig.update_geos(projection_type="natural earth", scope="europe")

fig.update_layout(

    title_text='Latest BCG reported Europe')
fig.update_geos(projection_type="natural earth", scope="africa")

fig.update_layout(title_text='Latest BCG reported Africa')
fig.update_geos(projection_type="natural earth", scope="south america")

fig.update_layout(title_text='Latest BCG reported South America')