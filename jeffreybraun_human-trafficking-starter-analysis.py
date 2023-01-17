import numpy as np
import pandas as pd
import seaborn as sns
import plotly as py
import plotly_express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import folium
from folium import plugins
from plotly.offline import init_notebook_mode, iplot
import os
init_notebook_mode()

!pip install --quiet pycountry_convert
from pycountry_convert import country_alpha2_to_country_name, country_name_to_country_alpha3

df = pd.read_csv('/kaggle/input/global-human-trafficking/human_trafficking.csv')
df.replace('-99', np.nan, inplace=True)
df.replace(-99, np.nan, inplace=True)
df.drop(columns = [df.columns[0]], inplace=True)
df.replace('00', np.nan, inplace=True)

print("Number of Victims of Unknown Citizenship: ")
num_null = len(df[df.citizenship.isnull()])
print(str(num_null) +" ({:.2%})".format(num_null / len(df)))


df_origin = df.copy()
df_origin.dropna(subset = ['citizenship'], inplace=True)
df_origin['citizenship_iso3'] = df_origin['citizenship'].apply(lambda x: country_name_to_country_alpha3(country_alpha2_to_country_name(x)))
df_origin['citizenship_name'] = df_origin['citizenship'].apply(lambda x: country_alpha2_to_country_name(x))

df_map1 = pd.DataFrame(df_origin.groupby(['citizenship_iso3', 'citizenship_name']).size()).reset_index()
df_map1.rename(columns = {0:'Number of Victims'}, inplace=True)

fig = px.choropleth(df_map1, locations="citizenship_iso3",
                    color="Number of Victims",
                    hover_name="citizenship_name",
                    title = 'Human Trafficking Victims by Citizenship',
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()

print("Number of Victims of Unknown Country Of Exploitation: ")
num_null = len(df[df.CountryOfExploitation.isnull()])
print(str(num_null) +" ({:.2%})".format(num_null / len(df)))


df_destination = df.copy()
df_destination.dropna(subset = ['CountryOfExploitation'], inplace=True)
df_destination['destination_iso3'] = df_destination['CountryOfExploitation'].apply(lambda x: country_name_to_country_alpha3(country_alpha2_to_country_name(x)))
df_destination['destination_name'] = df_destination['CountryOfExploitation'].apply(lambda x: country_alpha2_to_country_name(x))

df_map2 = pd.DataFrame(df_destination.groupby(['destination_iso3', 'destination_name']).size()).reset_index()
df_map2.rename(columns = {0:'Number of Victims'}, inplace=True)

fig = px.choropleth(df_map2, locations="destination_iso3",
                    color="Number of Victims",
                    hover_name="destination_name",
                    title = 'Human Trafficking Victims by Country of Exploitation',
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()

df_map1.rename(columns = {'citizenship_iso3':'iso3', 'citizenship_name':'name', 'Number of Victims':'Out'}, inplace = True)
df_map2.rename(columns = {'destination_iso3':'iso3', 'destination_name':'name','Number of Victims':'In'}, inplace = True)

df_flow = pd.merge(df_map1, df_map2,how='outer', on=['iso3','name'])
df_flow.replace(np.nan, 0, inplace=True)
df_flow['Human Trafficking Flow'] = df_flow['Out'] - df_flow['In']
df_flow['Human Trafficking Flow (Symmetric Log Scale)'] = np.log(np.abs(df_flow['Out'] - df_flow['In'])) * np.sign(df_flow['Out'] - df_flow['In'])
print("Positive Flow: Origin of Human Trafficking")
print("Negative Flow: Destination of Human Trafficking")

fig = px.choropleth(df_flow, locations="iso3",
                    color="Human Trafficking Flow (Symmetric Log Scale)",
                    hover_name="name",
                    title = 'Human Trafficking Flow',
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show()

print("Top 10 Origin Countries, by Total Victims: ")
print(df_flow.sort_values(by = ['Out'], ascending = False)[0:10][['name', 'Out']])
print('\n')

print("Top 10 Origin Countries, by Flow: ")
print(df_flow.sort_values(by = ['Human Trafficking Flow'], ascending = False)[0:10][['name', 'Human Trafficking Flow']])
print('\n')

print("Top 10 Destination Countries, by Total Victims: ")
print(df_flow.sort_values(by = ['In'], ascending = False)[0:10][['name', 'In']])
print('\n')

print("Top 10 Destination Countries, by Flow: ")
print(df_flow.sort_values(by = ['Human Trafficking Flow'], ascending = True)[0:10][['name', 'Human Trafficking Flow']])
print('\n')


df_age = df.groupby(['gender', 'ageBroad']).size().reset_index()
df_age.rename(columns = {0:'Number of Trafficked Individuals'}, inplace=True)

fig = px.pie(df_age.groupby('gender').sum().reset_index(), values = 'Number of Trafficked Individuals', names = 'gender', title = 'Gender of Human Trafficking Victims')
fig.show()

fig = px.bar(df_age, x = 'ageBroad', y = 'Number of Trafficked Individuals', color = 'gender',
            category_orders = {'ageBroad': ['0--8', '9--17', '18--20', '21--23', '24--26', '27--29', '30--38', '39--47', '48+']})
fig.show()
