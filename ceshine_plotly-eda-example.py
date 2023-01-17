import pandas as pd

import plotly.graph_objects as go

import plotly.express as px

# Plotly installation: https://plot.ly/python/getting-started/#jupyterlab-support-python-35
df = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_train.csv')

df=df[df["Date"]>"2020-03-09"] # Only keep dates with confirmed cases

df.head()
# Reference: https://plot.ly/python/time-series/

fig = go.Figure(

    [go.Scatter(x=df['Date'], y=df['ConfirmedCases'])],

    layout_title_text="Confirmed Cases in California"

)

fig.update_layout(

    yaxis_type="log",

    margin=dict(l=20, r=20, t=50, b=20),

    template="plotly_white")

fig.show()
df_test = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')

print(df_test.shape)

df_test.head()
public_leaderboard_start_date = "2020-03-12"

last_public_leaderboard_train_date = "2020-03-11"

public_leaderboard_end_date  = "2020-03-26"



cases  = df[df["Date"]==last_public_leaderboard_train_date]["ConfirmedCases"].values[0] * (2**(1/4))

df_test.insert(1, "ConfirmedCases", 0)
for i in range(15):

    df_test.loc[i, "ConfirmedCases"] = cases

    cases = cases * (2**(1/4))    

df_test.head()
# Reference: https://plot.ly/python/time-series/

fig = go.Figure(

    [

        go.Scatter(x=df['Date'], y=df['ConfirmedCases'], name="actual"),

        go.Scatter(x=df_test['Date'].iloc[:15], y=df_test['ConfirmedCases'], name="predicted"),

    ],

    layout_title_text="Confirmed Cases in California"

)

fig.update_layout(

    yaxis_type="log",

    margin=dict(l=20, r=20, t=50, b=20),

    template="plotly_white")

fig.show()
df_growth = pd.DataFrame({

    "Date": df["Date"].iloc[1:].values,

    "Rate": df["ConfirmedCases"].iloc[1:].values / df["ConfirmedCases"].iloc[:-1].values * 100

})
# Reference: https://plot.ly/python/bar-charts/

fig = px.bar(df_growth, x='Date', y='Rate', width=600, height=400)

fig.update_layout(

    margin=dict(l=20, r=20, t=50, b=20),

    template="plotly_white",

    title="Empirical Growth Rate",

    yaxis_title="Rate (%)"

)

fig.update_yaxes(range=[100, 135])

fig.show()
# rate = df_growth["Rate"].mean() / 100

rate = df_growth["Rate"].iloc[:7].median() / 100

print(f"Rate used: {rate:.4f}")

df_test = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')

public_leaderboard_start_date = "2020-03-12"

last_public_leaderboard_train_date = "2020-03-11"

public_leaderboard_end_date  = "2020-03-26"



cases  = df[df["Date"]==last_public_leaderboard_train_date]["ConfirmedCases"].values[0] * (rate)

df_test.insert(1, "ConfirmedCases", 0)

for i in range(15):

    df_test.loc[i, "ConfirmedCases"] = cases

    cases = cases * rate  

df_test.head()
# Reference: https://plot.ly/python/time-series/

fig = go.Figure(

    [

        go.Scatter(x=df['Date'], y=df['ConfirmedCases'], name="actual"),

        go.Scatter(x=df_test['Date'].iloc[:15], y=df_test['ConfirmedCases'], name="predicted"),

    ],

    layout_title_text="Confirmed Cases in California"

)

fig.update_layout(

    yaxis_type="log",

    margin=dict(l=20, r=20, t=50, b=20),

    template="plotly_white")

fig.show()
# Reference: https://plot.ly/python/time-series/

fig = go.Figure(

    [go.Scatter(x=df['Date'], y=df['Fatalities'])],

    layout_title_text="Fatalities in California"

)

fig.update_layout(

    yaxis_type="log",

    margin=dict(l=20, r=20, t=50, b=20),

    template="plotly_white")

fig.show()
df_growth = pd.DataFrame({

    "Date": df["Date"].iloc[1:].values,

    "Rate": df["Fatalities"].iloc[1:].values / df["Fatalities"].iloc[:-1].values * 100

})

# Reference: https://plot.ly/python/bar-charts/

fig = px.bar(df_growth, x='Date', y='Rate', width=600, height=400)

fig.update_layout(

    margin=dict(l=20, r=20, t=50, b=20),

    template="plotly_white",

    title="Empirical Growth Rate",

    yaxis_title="Rate (%)"

)

fig.update_yaxes(range=[100, 180])

fig.show()
rate = df_growth["Rate"].iloc[:7].median() / 100

print(f"Rate used: {rate:.4f}")

df_test = pd.read_csv('../input/covid19-local-us-ca-forecasting-week-1/ca_test.csv')

public_leaderboard_start_date = "2020-03-12"

last_public_leaderboard_train_date = "2020-03-11"

public_leaderboard_end_date  = "2020-03-26"



cases  = df[df["Date"]==last_public_leaderboard_train_date]["Fatalities"].values[0] * (rate)

df_test.insert(1, "Fatalities", 0)

for i in range(15):

    df_test.loc[i, "Fatalities"] = cases

    cases = cases * rate  

df_test.head()
# Reference: https://plot.ly/python/time-series/

fig = go.Figure(

    [

        go.Scatter(x=df['Date'], y=df['Fatalities'], name="actual"),

        go.Scatter(x=df_test['Date'].iloc[:15], y=df_test['Fatalities'], name="predicted"),

    ],

    layout_title_text="Fatalities in California"

)

fig.update_layout(

    yaxis_type="log",

    margin=dict(l=20, r=20, t=50, b=20),

    template="plotly_white")

fig.show()