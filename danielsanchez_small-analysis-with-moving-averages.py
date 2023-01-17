import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objects as go

#import plotly.plotly as py

#import plotly.figure_factory as ff

#from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from plotly.offline import iplot, init_notebook_mode

#from plotly import tools

init_notebook_mode(connected=True)
df_train = pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

df_test = pd.read_csv("../input/covid19-global-forecasting-week-3/test.csv")
df_train.head()
# Get countries where the data is divided in regions

tmp = df_train.groupby(["Country_Region", "Date"])["Province_State"].agg("count").reset_index()

countries_with_regions = tmp[tmp["Province_State"] > 1]["Country_Region"].unique()
countries_with_regions
def compute_country_data(country, num_cases = 100):

    

    df_country = df_train[df_train["Country_Region"] == country]

    

    if country in countries_with_regions:

        print(country, " has regions")

        df_country = df_country.groupby("Date")["ConfirmedCases", "Fatalities"].sum().reset_index()

    else:

        df_country = df_country[["Date", "ConfirmedCases", "Fatalities"]]

        



    df_country = df_country.sort_values("Date")

    

    # Compute new cases and new deaths          

    df_country["new_cases"] = df_country['ConfirmedCases'].diff(periods=1)

    df_country["new_deaths"] = df_country['Fatalities'].diff(periods=1)

    

    # Compute shift of cases. We know that symptoms have some delay to appear and positves are dected a few days they are infected

    df_country["ConfirmedCases_shift_10"] = df_country["ConfirmedCases"].shift(-10)

    df_country["ConfirmedCases_shift_7"] = df_country["ConfirmedCases"].shift(-7)

    df_country["new_cases_shift_10"] = df_country["new_cases"].shift(-10)

    df_country["new_cases_shift_7"] = df_country["new_cases"].shift(-7)

    

    # Compute number of days since num_cases

    days_since = pd.to_datetime(df_country[df_country["ConfirmedCases"] > num_cases].head(1)["Date"]).values[0]

    df_country["days_since"] = (pd.to_datetime(df_country["Date"]) - days_since).dt.days



    # Compute Moving Averages

    for i in [3,5,7,10]:

        df_country["new_cases_MA" + str(i)] = df_country["new_cases"].rolling(i, min_periods=i).mean().reset_index(0,drop=True).values

        df_country["new_deaths_MA"+ str(i)] = df_country["new_deaths"].rolling(i, min_periods=i).mean().reset_index(0,drop=True).values

        df_country["new_cases_shift_7_MA" + str(i)] = df_country["new_cases_shift_7"].rolling(i, min_periods=i).mean().reset_index(0,drop=True).values

        df_country["new_cases_shift_10_MA" + str(i)] = df_country["new_cases_shift_10"].rolling(i, min_periods=i).mean().reset_index(0,drop=True).values

        

    return df_country

df_china = compute_country_data("China")

df_korea = compute_country_data("Korea, South")

df_italy = compute_country_data("Italy")

df_spain = compute_country_data("Spain")

df_us = compute_country_data("US")

df_uk = compute_country_data("United Kingdom")
def double_line_plot(df, col1, col2, title=""):

    data = [go.Scatter(x=df["Date"], y=df[col1].values, name=col1),

            go.Scatter(x=df["Date"], y=df[col2].values, name=col2, yaxis='y2')]



    layout = go.Layout(dict(title = title, 

                        xaxis = dict(title = "Date",

                                     showgrid=False,

                                     zeroline=False,

                                     showline=False,),

                        yaxis = dict(title = "New Cases",

                                     showgrid=False,

                                     zeroline=False,

                                     showline=False,

                                     side='left'),

                        yaxis2=dict(title="New Deaths", overlaying='y', side='right')),

                   legend=dict(orientation="v"))

    return iplot(dict(data=data, layout=layout))



def double_bar_plot(df, col1, col2, title=""):

    fig = go.Figure(data=[

        go.Bar(x=df["Date"], y=df[col1].values, name=col1),

        go.Bar(x=df["Date"], y=df[col2].values, name=col2)

    ])

    # Change the bar mode

    fig.update_yaxes(title_text="<b>Number of accumulated cases (MA7)</b>")

    fig.update_layout(barmode='group', title_text=title)

    fig.show()



def double_bar_plot_2(df, col1, col2, title=""):

    data = [go.Bar(x=df["Date"], y=df[col1].values, name=col1),

            go.Bar(x=df["Date"], y=df[col2].values, name=col2, yaxis='y2')]



    layout = go.Layout(dict(title = title,

                            barmode='group',

                        xaxis = dict(title = "Date",

                                     showgrid=False,

                                     zeroline=False,

                                     showline=False,),

                        yaxis = dict(title = "New Cases",

                                     showgrid=False,

                                     zeroline=False,

                                     showline=False,

                                     side='left'),

                        yaxis2=dict(title="New Deaths", overlaying='y', side='right')),

                   legend=dict(orientation="v"))

    return iplot(dict(data=data, layout=layout))



#fig.update_layout(barmode='group')



def bar_line_plot(df, col1, col2, title=""):

    data = [go.Bar(x=df["Date"], y=df[col1].values, name=col1),

            go.Scatter(x=df["Date"], y=df[col2].values, name=col2, yaxis='y2')]



    layout = go.Layout(dict(title = title,

                        xaxis = dict(title = "Date",

                                     showgrid=False,

                                     zeroline=False,

                                     showline=False,),

                        yaxis = dict(title = "New Cases",

                                     showgrid=False,

                                     zeroline=False,

                                     showline=False,

                                     side='left'),

                        yaxis2=dict(title="New Deaths", overlaying='y', side='right')),

                   legend=dict(orientation="v"))

    return iplot(dict(data=data, layout=layout))
double_line_plot(df_china, "new_cases_MA7", "new_deaths_MA7", title="New Cases and Deaths in China")

double_line_plot(df_italy, "new_cases_MA7", "new_deaths_MA7", title="New Cases and Deaths in Italy")

double_line_plot(df_spain, "new_cases_MA7", "new_deaths_MA7", title="New Cases and Deaths in Spain")

double_line_plot(df_us, "new_cases_MA7", "new_deaths_MA7", title="New Cases and Deaths in US")
fig = go.Figure()



df_italy_tmp = df_italy[(df_italy["days_since"] > 0) & (df_italy["days_since"] < 40)]

df_spain_tmp = df_spain[(df_spain["days_since"] > 0) & (df_spain["days_since"] < 40)]

#df_us_tmp = df_us[(df_us["days_since"] > 0) & (df_us["days_since"] < 40)]

fig.add_trace(go.Scatter(x=df_italy_tmp["days_since"], y=df_italy_tmp["new_cases_MA7"], name="Italy"))

fig.add_trace(go.Scatter(x=df_spain_tmp["days_since"], y=df_spain_tmp["new_cases_MA7"], name="Spain"))

#fig.add_trace(go.Scatter(x=df_us_tmp["days_since"], y=df_us_tmp["new_cases_MA7"], name="USA"))



fig.update_xaxes(title_text="<b>Days since the 100th case</b>")

fig.update_yaxes(title_text="<b>Number of new cases (MA7)</b>", showgrid=False, zeroline=False,showline=False)

fig.update_layout(title_text="Number of cases since the day of the 100th case")

fig.show()
fig = go.Figure()



df_italy_tmp = df_italy[(df_italy["days_since"] > 0) & (df_italy["days_since"] < 40)]

df_spain_tmp = df_spain[(df_spain["days_since"] > 0) & (df_spain["days_since"] < 40)]

df_us_tmp = df_us[(df_us["days_since"] > 0) & (df_us["days_since"] < 40)]

fig.add_trace(go.Scatter(x=df_italy_tmp["days_since"], y=df_italy_tmp["new_deaths_MA7"], name="Deaths Italy"))

fig.add_trace(go.Scatter(x=df_spain_tmp["days_since"], y=df_spain_tmp["new_deaths_MA7"], name="Deaths Spain"))

fig.add_trace(go.Scatter(x=df_us_tmp["days_since"], y=df_us_tmp["new_deaths_MA7"], name="USA"))



fig.update_xaxes(title_text="<b>Days since the 100th case</b>")

fig.update_yaxes(title_text="<b>Number of new deaths (MA7)</b>", showgrid=False, zeroline=False,showline=False)

fig.update_layout(title_text="Number of daily deaths since the day of the 100th case")

fig.show()
df_italy_tmp = df_italy[(df_italy["Date"] > '2020-03-01')]

df_spain_tmp = df_spain[(df_spain["Date"] > '2020-03-01')]



fig = go.Figure(data=[

        go.Bar(x=df_italy_tmp["Date"], y=df_italy_tmp["Fatalities"].values, name="Italy"),

        go.Bar(x=df_spain_tmp["Date"], y=df_spain_tmp["Fatalities"].values, name="Spain")

    ])



fig.update_yaxes(title_text="<b>Number of accumulated deaths</b>")

fig.update_layout(barmode='group', title_text="Number of accumulated deaths in Spain and Italy")

fig.show()
df_italy_tmp = df_italy[(df_italy["days_since"] > 0) & (df_italy["days_since"] < 35)]

df_spain_tmp = df_spain[(df_spain["days_since"] > 0) & (df_spain["days_since"] < 35)]



fig = go.Figure(data=[

        go.Bar(x=df_italy_tmp["days_since"], y=df_italy_tmp["Fatalities"].values, name="italy"),

        go.Bar(x=df_spain_tmp["days_since"], y=df_spain_tmp["Fatalities"].values, name="spain")

    ])



fig.update_yaxes(title_text="<b>Number of accumulated deaths</b>")

fig.update_layout(barmode='group', title_text="Number of accumulated deaths in Spain and Italy starting from the day of the 100th case")

fig.show()