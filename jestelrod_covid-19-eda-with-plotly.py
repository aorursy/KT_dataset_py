# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

country_codes = pd.read_csv("../input/countries-iso-codes/wikipedia-iso-country-codes.csv")

population_density = pd.read_csv("../input/population-density/population_density.csv", delimiter=";",error_bad_lines = False, index_col = "Country Code")

population_middle_age = pd.read_csv("../input/population-middle-age/Population_middle_age.csv", delimiter=",",error_bad_lines = False, index_col = "Place")
population_middle_age = population_middle_age.Median
Pop_dens = "2018"

population_density = population_density.loc[:,Pop_dens]
confirmed_df = pd.merge(confirmed_df, country_codes, right_on = "English short name lower case", left_on = "Country/Region")

confirmed_df = pd.merge(confirmed_df, population_density, right_index = True, left_on = "Alpha-3 code")

confirmed_df = pd.merge(confirmed_df, population_middle_age, right_index = True, left_on = "Country/Region")

deaths_df = pd.merge(deaths_df, country_codes, right_on = "English short name lower case", left_on = "Country/Region")

deaths_df = pd.merge(deaths_df, population_density, right_index = True, left_on = "Alpha-3 code")
cols = confirmed_df.keys()

Last_Date = cols[-8]

codes = cols[-5]

First_Date = cols[4]
confirmed_df = confirmed_df[confirmed_df[Last_Date]>=400]

deaths_df = deaths_df[deaths_df[Last_Date]>=4]
deaths_df = deaths_df.groupby("Country/Region").apply(lambda df: df.iloc[:,4:].sum()).sort_values(by=Last_Date)

confirmed_df = confirmed_df.groupby("Country/Region").apply(lambda df: df.iloc[:,4:].sum()).sort_values(by=Last_Date)
#df.replace({'A': {0: 100, 4: 400}})

#confirmed_df = confirmed_df.replace({"Alpha-3 code": {"China": "CHN", "Canada":"CAN", "Australia":"AUS"}})

#confirmed_df.replace({confirmed_df.at["China","Alpha-3 code"]:"CHN"})

#confirmed_df.tail(30)

#confirmed_df.at["China","Alpha-3 code"]

confirmed_df.loc[confirmed_df['Alpha-3 code'] ==confirmed_df.at["China","Alpha-3 code"], 'Alpha-3 code'] = 'CHN'

confirmed_df.loc[confirmed_df['Alpha-3 code'] ==confirmed_df.at["Canada","Alpha-3 code"], 'Alpha-3 code'] = 'CAN'

confirmed_df.loc[confirmed_df['Alpha-3 code'] ==confirmed_df.at["Australia","Alpha-3 code"], 'Alpha-3 code'] = 'AUS'

deaths_df.loc[deaths_df['Alpha-3 code'] ==deaths_df.at["Australia","Alpha-3 code"], 'Alpha-3 code'] = 'AUS'

deaths_df.loc[deaths_df['Alpha-3 code'] ==deaths_df.at["China","Alpha-3 code"], 'Alpha-3 code'] = 'CHN'

deaths_df.loc[deaths_df['Alpha-3 code'] ==deaths_df.at["Canada","Alpha-3 code"], 'Alpha-3 code'] = 'CAN'

rate_dead_df = pd.Series(deaths_df.loc[:,Last_Date]/confirmed_df.loc[:,Last_Date]*100).dropna().sort_values()
import plotly.express as px

fig = px.bar(confirmed_df, x = confirmed_df.index , y = Last_Date, color = Last_Date )

fig.show()

fig_dead = px.bar(deaths_df, x = deaths_df.index , y = Last_Date, color = Last_Date )

fig_dead.show()

fig_rate = px.bar(rate_dead_df, x = rate_dead_df.index, y = Last_Date, color = Last_Date)

fig_rate.show()
df_analisys = pd.DataFrame([confirmed_df["Alpha-3 code"],

                            confirmed_df[Last_Date],

                            deaths_df[Last_Date],

                            rate_dead_df,

                            confirmed_df.Median,

                            confirmed_df[Pop_dens]]).transpose().fillna(1)

df_analisys.columns = ["Country Codes" , "Confirmed", "Dead", "Rate Dead", "Median" ,"Population Density"]
fig_map = px.choropleth(df_analisys, locations = "Country Codes",

                       color = "Confirmed",

                       hover_name = df_analisys.index,

                       color_continuous_scale=px.colors.sequential.Plasma)

fig_dead_map = px.choropleth(df_analisys, locations = "Country Codes",

                       color = "Dead",

                       hover_name = df_analisys.index,

                       color_continuous_scale=px.colors.sequential.Plasma)

fig_ratedead_map = px.choropleth(df_analisys, locations = "Country Codes",

                       color = "Rate Dead",

                       hover_name = df_analisys.index,

                       color_continuous_scale=px.colors.sequential.Plasma)

fig_map.show()

fig_dead_map.show()

fig_ratedead_map.show()
fig_dense = px.scatter(df_analisys, x="Confirmed", y="Population Density",

                 size='Dead',hover_name = df_analisys.index, trendline ="ols")

fig_b = px.scatter(df_analisys, x = "Rate Dead", y= "Confirmed", size = "Dead", color = "Population Density",hover_name = df_analisys.index, trendline ="ols")

fig_dense.show()

fig_b.show()
df_analisys.loc[df_analisys['Median'] ==df_analisys.at["China","Median"], 'Median'] = 37.4

fig = px.scatter(df_analisys, x = "Median", y= "Rate Dead", size = "Dead", color = "Population Density",trendline ="ols", hover_name = df_analisys.index)

fig.show()

fig_b = px.scatter(df_analisys, x = "Median", y= "Confirmed", size = "Dead", color = "Population Density",trendline ="ols", hover_name = df_analisys.index)

fig_b.show()
Countries_ = ["Italy", "China", "Spain", "Iran", "Germany", "France"]

df_top_infec = confirmed_df.loc[Countries_,First_Date : Last_Date]

df_top_dead = deaths_df.loc[Countries_,First_Date : Last_Date]
import plotly.graph_objects as go



fig_count = go.Figure()

# Create and style traces

fig_count.add_trace(go.Scatter(x=df_top_infec.columns, y=df_top_infec.loc[Countries_[0]], name=Countries_[0],

                            line=dict(color='Green', width=4)))

fig_count.add_trace(go.Scatter(x=df_top_infec.columns, y=df_top_infec.loc[Countries_[1]], name=Countries_[1],

                            line=dict(color='Yellow', width=4)))

fig_count.add_trace(go.Scatter(x=df_top_infec.columns, y=df_top_infec.loc[Countries_[2]], name=Countries_[2],

                            line=dict(color='Red', width=4)))

fig_count.add_trace(go.Scatter(x=df_top_infec.columns, y=df_top_infec.loc[Countries_[3]], name=Countries_[3],

                            line=dict(color='Brown', width=4)))

fig_count.add_trace(go.Scatter(x=df_top_infec.columns, y=df_top_infec.loc[Countries_[4]], name=Countries_[4],

                            line=dict(color='Black', width=4)))

fig_count.add_trace(go.Scatter(x=df_top_infec.columns, y=df_top_infec.loc[Countries_[5]], name=Countries_[5],

                            line=dict(color='Blue', width=4)))

fig_count.show()
fig_count = go.Figure()

# Create and style traces

fig_count.add_trace(go.Scatter(x=df_top_dead.columns, y=df_top_dead.loc[Countries_[0]], name=Countries_[0],

                            line=dict(color='Green', width=4)))

fig_count.add_trace(go.Scatter(x=df_top_dead.columns, y=df_top_dead.loc[Countries_[1]], name=Countries_[1],

                            line=dict(color='Yellow', width=4)))

fig_count.add_trace(go.Scatter(x=df_top_dead.columns, y=df_top_dead.loc[Countries_[2]], name=Countries_[2],

                            line=dict(color='Red', width=4)))

fig_count.add_trace(go.Scatter(x=df_top_dead.columns, y=df_top_dead.loc[Countries_[3]], name=Countries_[3],

                            line=dict(color='Brown', width=4)))

fig_count.add_trace(go.Scatter(x=df_top_dead.columns, y=df_top_dead.loc[Countries_[4]], name=Countries_[4],

                            line=dict(color='Black', width=4)))

fig_count.add_trace(go.Scatter(x=df_top_dead.columns, y=df_top_dead.loc[Countries_[5]], name=Countries_[5],

                            line=dict(color='Blue', width=4)))

fig_count.show()
df_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv", index_col = "id",)

df_data_open = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv", parse_dates = ["date_admission_hospital","date_death_or_discharge"])
def correct_sex(x):

    if x == "Male" or x == "male":

        return "Male"

    elif x == "female" or x == "Female":

        return "Female"

    else:

        return np.nan

df_gender = df_data_open.sex.apply(correct_sex).value_counts()

import plotly.express as px

fig = px.bar(df_gender, x = df_gender.index ,y = "sex", color = "sex")

fig.show()
def correct_age(row):

    try:

        y = int(row)

        return y

    except:

        return np.nan

df_age = df_data_open.age.apply(correct_age)
def correct_outcome(x):

    if x == "discharged" or x == "discharge" or x == "Discharged":

        return "Discharged"

    elif x == "died" or x == "death":

        return "Death"

    elif x == "recovered" or x =="stable":

        return "Recovered"

    else:

        return np.nan

df_outcome = df_data_open.outcome.apply(correct_outcome)
df_ = pd.DataFrame([df_age,

                    df_outcome ,

                    df_data_open.date_admission_hospital, 

                    df_data_open.date_death_or_discharge,

                    df_data_open.country],).transpose()
df_.dropna(subset = ["age"])

df_country_ =df_.groupby("country").country.count()
df_outcome_ = df_.groupby("outcome").outcome.count()

fig = px.pie(df_outcome_, names = df_outcome_.index,values = "outcome", title = "Situation of people infected by COVID-19")

fig.show()
df_dn_outcome = df_.dropna(subset = ["date_death_or_discharge"])
def group_age(x):

    if x >= 90:

        return 10

    elif x >=80:

        return 9

    elif x >=70:

        return 8

    elif x >=60:

        return 7

    elif x >=50:

        return 6

    elif x >=40:

        return 5

    elif x >=30:

        return 4

    elif x >=20:

        return 3

    elif x >=10:

        return 2

    elif x >=0:

        return 1

df_age = df_.dropna(subset = ["age"])

df_age_ = df_age.age.apply(group_age).value_counts()

fig = px.bar(df_age_, x = df_age_.index,y = "age")

fig.show()