import numpy as np

import pandas as pd 

import plotly.express as px
df = pd.read_csv('/kaggle/input/percent-black-population-for-every-state-in-usa/percent_black_over_time.csv')

df = df[:1]

cols_to_check = ['1790', '1800', '1810', '1820', '1830', '1840',

       '1850', '1860', '1870', '1880', '1890', '1900', '1910', '1920',

       '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000',

       '2010', '2018']

df[cols_to_check] = df[cols_to_check].replace({'%':''}, regex=True)

df = df.transpose()

new_header = df.iloc[0]

df = df[1:]

df.columns = new_header

df['Year'] = df.index

df.columns = ['Percent Black','Year']

df = df.astype(float)

df.to_csv('/kaggle/working/percent_black_in_usa.csv',index=False)
plot = px.line(df, 

               x="Year", 

               y="Percent Black", 

               hover_name="Percent Black",

               title='Percentage of Population in USA that Identifies as Black',

               line_shape="spline")

plot
df = pd.read_csv('/kaggle/input/percent-black-population-for-every-state-in-usa/percent_black_over_time.csv')

df = df[1:]

cols_to_check = ['1790', '1800', '1810', '1820', '1830', '1840',

       '1850', '1860', '1870', '1880', '1890', '1900', '1910', '1920',

       '1930', '1940', '1950', '1960', '1970', '1980', '1990', '2000',

       '2010', '2018']

df[cols_to_check] = df[cols_to_check].replace({'%':''}, regex=True)

df = df.transpose()

new_header = df.iloc[0]

df = df[1:].fillna(0)

df.columns = new_header

df = df.astype(float)

df = df[-1:]

df = df.transpose()

df.columns = ['Percent Black']

df['State/Territory'] = df.index

df.to_csv('/kaggle/working/percent_black_in_state_by_state.csv',index=False) # save to notebook output
plot = px.bar(df, x=df.index, y="Percent Black", hover_name="Percent Black",title='Percentage of Population in USA that Identifies as Black (in 2018)') 

plot
df = pd.read_csv('/kaggle/input/percent-black-population-for-every-state-in-usa/percent_black_in_2018.csv')

fig = px.choropleth(df, 

                    locations="State Code", 

                    color="Percent Black", 

                    locationmode = 'USA-states', 

                    hover_name="State",

                    range_color=[0,30],scope="usa",

                    title='Percent of Population that Identifies as Black (2018)')

fig.show()