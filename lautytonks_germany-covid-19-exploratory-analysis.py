## import libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

%matplotlib inline

import warnings

warnings.filterwarnings("ignore")





## import data



cov = pd.read_csv('/kaggle/input/covid19-tracking-germany/covid_de.csv')

demo = pd.read_csv('/kaggle/input/covid19-tracking-germany/demographics_de.csv')
## covid-19 Germany cases per state



cov.head(10)
## general demographic information in germany states



demo.head(10)
cov.dtypes

#convert date to datetime for time series analysis 



d = cov["date"]

cov["date"] = pd.to_datetime(d)
demo.dtypes
gender_age = demo.groupby(['gender', 'age_group']).sum().reset_index()
gender_age.dtypes





fig = px.bar(gender_age, y='population', x='gender', color='age_group')

fig.update_layout(title='distribution of ages per gender')

fig.show()



state_age = demo.groupby(['state', 'age_group']).sum().reset_index()




fig = px.bar(state_age, y='population', x='state', color='age_group')

fig.update_layout(title='distribution of ages per state')

fig.show()



cov_cases = cov.groupby(['date']).sum().reset_index()
fig, ax = plt.subplots(figsize=(16,9))

ax.plot(cov_cases["date"],

        cov_cases["cases"],

        color="g");

ax.set_title("germany confirmed cases per day");

ax.spines["top"].set_visible(False);

ax.spines["right"].set_visible(False);
fig, ax = plt.subplots(figsize=(16,9))

ax.plot(cov_cases["date"],

        cov_cases["deaths"],

        color="r");

ax.set_title("germany confirmed deaths per day");

ax.spines["top"].set_visible(False);

ax.spines["right"].set_visible(False);
fig, ax = plt.subplots(figsize=(16,9))

ax.plot(cov_cases["date"],

        cov_cases["recovered"],

        color="b");

ax.set_title("germany recovered cases per day");

ax.spines["top"].set_visible(False);

ax.spines["right"].set_visible(False);
## group by gender 



cov_gen = cov.groupby(['gender']).sum().reset_index()
## calculate percentages for more comparable results



cov_gen["death_percentage"] = round(cov_gen["deaths"]/cov_gen["cases"] * 100,0)

cov_gen["cases_recovering"] = round(cov_gen["recovered"]/cov_gen["cases"] * 100,0)





cov_gen
cov_age = cov.groupby(['age_group']).sum().reset_index()


cov_age["death_percentage"] = round(cov_age["deaths"]/cov_age["cases"]* 100,1)
cov_age
cov_age_gen = cov.groupby(['age_group','gender']).sum().reset_index()


cov_age_gen["death_percentage"] = round(cov_age_gen["deaths"]/cov_age_gen["cases"] * 100,1)

cov_age_gen["cases_per_age"] = round(cov_age_gen["cases"]/cov_age_gen["cases"].sum() * 100,1)

cov_age_gen

state_df = cov.groupby(['state','date']).sum().reset_index()
state_df["state"].unique()
state_df["state"] = state_df["state"].str.replace("Baden-Wuerttemberg","Baden").str.replace("Mecklenburg-Vorpommern","Mecklenburg").str.replace("Nordrhein-Westfalen","Nordrhein").str.replace("Sachsen-Anhalt","Sachsen_A").str.replace("Schleswig-Holstein","Schleswig").str.replace("Rheinland-Pfalz","Rheinland")

listofstates = state_df["state"].unique()



## create for loop to split dfs by state for analysis 



listofdfs = []



for state in listofstates:

    locals()['df_' + state] = state_df[(state_df.state== state)]

    listofdfs.append(['df_'+ state][0])
listofdfs = [ df_Baden,

              df_Bayern,

              df_Berlin,

              df_Brandenburg,

              df_Bremen,

              df_Hamburg,

              df_Hessen,

              df_Mecklenburg,

              df_Niedersachsen,

              df_Nordrhein,

              df_Rheinland,

              df_Saarland,

              df_Sachsen,

              df_Sachsen_A,

              df_Schleswig,

              df_Thueringen]
def cumsum(df):

    df['cumsum_deaths'] = df["deaths"].cumsum()

    df['cumsum_cases'] = df["cases"].cumsum()

    df['cumsum_recovered'] = df["recovered"].cumsum()
for i in listofdfs:

    cumsum(i)
## merge df with cumsum figures



merged_df = pd.concat([df_Baden,

                            df_Bayern,

                            df_Berlin,

                            df_Brandenburg,

                            df_Bremen,

                            df_Hamburg,

                            df_Hessen,

                            df_Mecklenburg,

                            df_Niedersachsen,

                            df_Nordrhein,

                            df_Rheinland,

                            df_Saarland,

                            df_Sachsen,

                            df_Sachsen_A,

                            df_Schleswig,

                            df_Thueringen], axis=0)
merged_df




fig = px.line(merged_df,

              x="date",

              y="cumsum_cases",

              color="state",

              line_group="state",

              hover_name="state")

fig.update_layout(

              title="cumulative cases per state",

              yaxis_title="cumulative_cases")



fig.show()
## create for loop for multiple graphs per state



for i in listofdfs:



    fig = px.line(i,

                  x="date",

                  y="cumsum_cases",

                  color="state",

                  line_group="state",

                  hover_name="state")

    fig.update_layout(

                  title="cumulative cases per state",

                  yaxis_title="cumulative_cases",)

    fig.show()




fig = px.line(merged_df,

              x="date",

              y="cumsum_deaths",

              color="state",

              line_group="state",

              hover_name="state")

fig.update_layout(

              title="cumulative deaths per state",

              yaxis_title="cumulative_deaths"

)

fig.show()
## create for loop for multiple graphs per state



for i in listofdfs:



    fig = px.line(i,

                  x="date",

                  y="cumsum_deaths",

                  color="state",

                  line_group="state",

                  hover_name="state")

    fig.update_layout(

                  title="cumulative deaths per state",

                  yaxis_title="cumulative_deaths",)

    fig.show()




fig = px.line(merged_df,

              x="date",

              y="cumsum_recovered",

              color="state",

              hover_name="state")

fig.update_layout(

              title="cumulative recoveries per state",

              yaxis_title="cumulative_recoveries"

          )



fig.show()
## create for loop for multiple graphs per state



for i in listofdfs:



    fig = px.line(i,

                  x="date",

                  y="cumsum_recovered",

                  color="state",

                  hover_name="state")

    fig.update_layout(

                  title="cumulative recoveries per state",

                  yaxis_title="cumulative_recoveries",)

    fig.show()