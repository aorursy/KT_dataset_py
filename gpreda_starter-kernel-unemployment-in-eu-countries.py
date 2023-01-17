import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime as dt

import folium

from folium.plugins import HeatMap, HeatMapWithTime

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")
data_df = pd.read_csv(os.path.join("/kaggle", "input", "unemployment-in-european-union", "une_rt_m.tsv"), sep='\t')
country_codes_df = pd.read_csv(os.path.join("/kaggle", "input", "iso-country-codes-global", "wikipedia-iso-country-codes.csv"))
data_df.shape
data_df.head()
data_df.columns
country_codes_df.head()
country_codes_df.columns = ['country', 'C2', 'C3', 'numeric', 'iso']
country_codes_df.head()
data_df['C2'] = data_df['s_adj,age,unit,sex,geo\\time'].apply(lambda x: x.split(",")[-1])

data_df['age'] = data_df['s_adj,age,unit,sex,geo\\time'].apply(lambda x: x.split(",")[1])

data_df['unit'] = data_df['s_adj,age,unit,sex,geo\\time'].apply(lambda x: x.split(",")[2])

data_df['sex'] = data_df['s_adj,age,unit,sex,geo\\time'].apply(lambda x: x.split(",")[3])

data_df['s_adj'] = data_df['s_adj,age,unit,sex,geo\\time'].apply(lambda x: x.split(",")[0])
data_df.head()
print(f"countries:\n{list(data_df.C2.unique())}")
print(f"sex:\n{list(data_df.sex.unique())}")
print(f"age intervals:\n{list(data_df.age.unique())}")
print(f"unit:\n{list(data_df.unit.unique())}")
print(f"s_adj:\n{list(data_df.s_adj.unique())}")
selected_cols = ['C2','age','unit','sex', 's_adj', 

                 '2020M07 ', '2020M06 ', '2020M05 ', '2020M04 ','2020M03 ','2020M02 ','2020M01 ',

                 '2019M12 ','2019M11 ','2019M10 ','2019M09 ','2019M08 ','2019M07 ',

                 '2019M06 ','2019M05 ','2019M04 ','2019M03 ','2019M02 ','2019M01 ',

                '2018M12 ','2018M11 ','2018M10 ','2018M09 ','2018M08 ','2018M07 ',

                 '2018M06 ','2018M05 ','2018M04 ','2018M03 ','2018M02 ','2018M01 ',

                '2017M12 ','2017M11 ','2017M10 ','2017M09 ','2017M08 ','2017M07 ',

                 '2017M06 ','2017M05 ','2017M04 ','2017M03 ','2017M02 ','2017M01 ',

                '2016M12 ','2016M11 ','2016M10 ','2016M09 ','2016M08 ','2016M07 ',

                 '2016M06 ','2016M05 ','2016M04 ','2016M03 ','2016M02 ','2016M01 ',

                '2015M12 ','2015M11 ','2015M10 ','2015M09 ','2015M08 ','2015M07 ',

                 '2015M06 ','2015M05 ','2015M04 ','2015M03 ','2015M02 ','2015M01 ']
data_sel_df = data_df[selected_cols]
data_sel_df = data_sel_df.merge(country_codes_df, on="C2")
data_sel_df.head()
print(f"selected data shape: {data_sel_df.shape}")
data_tr_df = data_sel_df.melt(id_vars=["country", "age", "unit", "sex", "s_adj", "C2", "C3", "numeric", "iso"], 

        var_name="Date", 

        value_name="Value")
data_tr_df.head()
print(f"new data shape: {data_tr_df.shape}")
import re

data_tr_df['Value'] = data_tr_df['Value'].apply(lambda x: re.sub(r"[a-zA-Z: ]", "", x))

data_tr_df['Value'] = data_tr_df['Value'].apply(lambda x: x.replace(" ",""))



data_tr_df = data_tr_df.loc[~(data_tr_df.Value=="")]



data_tr_df['Value'] = data_tr_df['Value'].apply(lambda x: float(x))
print(f"distinct values: {len(list(data_tr_df['Value'].unique()))}")

print(f"samples values: {data_tr_df['Value'].unique()}")
total_y25_74_df = data_tr_df.loc[(data_tr_df.age=='Y25-74')&(data_tr_df.unit=='PC_ACT')&(data_tr_df.sex=='T')&(data_tr_df.s_adj=='TC')]
def plot_time_variation(df, y='Value', size=1, is_log=False, title=""):

    f, ax = plt.subplots(1,1, figsize=(4*size,3*size))



    countries = list(df.country.unique())

    for country in countries:

        df_ = df[(df['country']==country)] 

        g = sns.lineplot(x="Date", y=y, data=df_,  label=country)  

        ax.text(max(df_['Date']), (df_.loc[df_['Date']==max(df_['Date']), y]), str(country))

    plt.xticks(rotation=90)

    plt.title(f'Total unemployment, {title}, grouped by country')

    ax.text(max(df_['Date']), (df_.loc[df_['Date']==max(df_['Date']), y]), str(country))

    plt.legend(loc="upper left", bbox_to_anchor=(1,1))

    if(is_log):

        ax.set(yscale="log")

    ax.grid(color='black', linestyle='dotted', linewidth=0.75)

    plt.show()  

plot_time_variation(total_y25_74_df, size=4, is_log=True, title = "age group 24-75 -")
total_F_y25_74_df = data_tr_df.loc[(data_tr_df.age=='Y25-74')&(data_tr_df.unit=='PC_ACT')&(data_tr_df.sex=='F')&(data_tr_df.s_adj=='TC')]

plot_time_variation(total_F_y25_74_df, size=4, is_log=True, title = "female, age group 24-75 ")
total_M_y25_74_df = data_tr_df.loc[(data_tr_df.age=='Y25-74')&(data_tr_df.unit=='PC_ACT')&(data_tr_df.sex=='M')&(data_tr_df.s_adj=='TC')]

plot_time_variation(total_M_y25_74_df, size=4, is_log=True, title = "male, age group 24-75 ")
total_M_y25_df = data_tr_df.loc[(data_tr_df.age=='Y_LT25')&(data_tr_df.unit=='PC_ACT')&(data_tr_df.sex=='M')&(data_tr_df.s_adj=='TC')]

plot_time_variation(total_M_y25_df, size=4, is_log=True, title = "male, age group <25 ")
total_F_y25_df = data_tr_df.loc[(data_tr_df.age=='Y_LT25')&(data_tr_df.unit=='PC_ACT')&(data_tr_df.sex=='F')&(data_tr_df.s_adj=='TC')]

plot_time_variation(total_F_y25_df, size=4, is_log=True, title = "female, age group <25 ")
def plot_time_variation_age_sex(data_tr_df, y='Value', country="Netherlands"):

    c_df = data_tr_df.loc[(data_tr_df.country==country)&(data_tr_df.unit=='PC_ACT')&(data_tr_df.s_adj=='TC')]

    f, ax = plt.subplots(1,1, figsize=(16,12))

    sns.lineplot(x="Date", y=y, data=c_df.loc[(c_df.age=='Y_LT25')&(c_df.sex=='F')],  label="Female, <25y")  

    sns.lineplot(x="Date", y=y, data=c_df.loc[(c_df.age=='Y_LT25')&(c_df.sex=='M')],  label="Male, <25y")  

    sns.lineplot(x="Date", y=y, data=c_df.loc[(c_df.age=='Y25-74')&(c_df.sex=='F')],  label="Female, 25-74y")  

    sns.lineplot(x="Date", y=y, data=c_df.loc[(c_df.age=='Y25-74')&(c_df.sex=='M')],  label="Male, <25-74y")  



    plt.xticks(rotation=90)

    plt.title(f'Total unemployment in {country}, grouped by age & sex')

    plt.legend(loc="upper left", bbox_to_anchor=(1,1))

    ax.grid(color='black', linestyle='dotted', linewidth=0.75)

    plt.show()  
plot_time_variation_age_sex(data_tr_df,country="Netherlands")
plot_time_variation_age_sex(data_tr_df,country="Denmark")
plot_time_variation_age_sex(data_tr_df,country="Sweden")
plot_time_variation_age_sex(data_tr_df,country="Estonia")
plot_time_variation_age_sex(data_tr_df,country="Latvia")
plot_time_variation_age_sex(data_tr_df,country="Lithuania")
plot_time_variation_age_sex(data_tr_df,country="Romania")
import plotly.express as px



def plot_animated_map(dd_df, title):

    hover_text = []

    for index, row in dd_df.iterrows():

        hover_text.append((f"country: {row['country']}<br>unemployment: {row['Value']}%<br>country code: {row['iso']}"))

    dd_df['hover_text'] = hover_text



    fig = px.choropleth(dd_df, 

                        locations="C3",

                        hover_name='hover_text',

                        color="Value",

                        animation_frame="Date",

                        projection="natural earth",

                        color_continuous_scale=px.colors.sequential.Plasma,

                        width=600, height=600)

    fig.update_geos(   

        showcoastlines=True, coastlinecolor="DarkBlue",

        showland=True, landcolor="LightGrey",

        showocean=True, oceancolor="LightBlue",

        showlakes=True, lakecolor="Blue",

        showrivers=True, rivercolor="Blue",

        showcountries=True, countrycolor="DarkBlue"

    )

    fig.update_layout(title = title, geo_scope="europe")

    fig.show()    
c_df = data_tr_df.loc[(data_tr_df.unit=='PC_ACT')&(data_tr_df.s_adj=='TC')]

dd_df=c_df.loc[(c_df.age=='Y_LT25')&(c_df.sex=='F')]

dd_df = dd_df.sort_values(by='Date')

title = 'Percent of unemployed per country<br>Female, under 25 - (hover for details)'

plot_animated_map(dd_df, title)
c_df = data_tr_df.loc[(data_tr_df.unit=='PC_ACT')&(data_tr_df.s_adj=='TC')]

dd_df=c_df.loc[(c_df.age=='Y_LT25')&(c_df.sex=='M')]

dd_df = dd_df.sort_values(by='Date')

title = 'Percent of unemployed per country<br>Male, under 25 - (hover for details)'

plot_animated_map(dd_df, title)
c_df = data_tr_df.loc[(data_tr_df.unit=='PC_ACT')&(data_tr_df.s_adj=='TC')]

dd_df=c_df.loc[(c_df.age=='Y25-74')&(c_df.sex=='F')]

dd_df = dd_df.sort_values(by='Date')

title = 'Percent of unemployed per country<br>Female, 25-74 yrs. old - (hover for details)'

plot_animated_map(dd_df, title)
c_df = data_tr_df.loc[(data_tr_df.unit=='PC_ACT')&(data_tr_df.s_adj=='TC')]

dd_df=c_df.loc[(c_df.age=='Y25-74')&(c_df.sex=='M')]

dd_df = dd_df.sort_values(by='Date')

title = 'Percent of unemployed per country<br>Male, 25-74 yrs. old - (hover for details)'

plot_animated_map(dd_df, title)