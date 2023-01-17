import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
df = pd.read_csv("../input/us-electoral-college-votes-per-state-17882020/Electoral_College.csv")
state_code = pd.read_csv("../input/two-letter-us-state-codes/StateCode.csv")
df.head(10)
total_votes = df.groupby("Year")["Votes"].sum().to_dict()
total_states = df.dropna().groupby("Year").count()['State'].to_dict()
def calc_prop(row):
    if row['Votes'] != np.nan:
        prop = row['Votes']/total_votes[row["Year"]] * 100
        return prop
    return np.nan

df['VoteProp'] = df.apply(calc_prop, axis = 1)
map_state = dict(zip(list(state_code['State'].values),
                     list(state_code['Code'].values)))
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Bar(x = list(total_votes.keys()), 
                         y = list(total_states.values()),
                         name = 'Total number of States in Electoral College over time',
                         opacity = 0.6),
                         secondary_y=False)

fig.add_trace(go.Scatter(x = list(total_votes.keys()), 
                         y = list(total_votes.values()),
                         mode='lines',
                         name = 'Total number of Electoral College votes over time'),
                         secondary_y = True)

fig.update_layout(
    title = "<b>Number of States in and Votes of Electoral College over time</b>",
    xaxis_title = "Year",
    plot_bgcolor='white')
px.choropleth(data_frame = df,
              locations= df['State'].map(map_state),
              color = df['VoteProp'],
              locationmode='USA-states',
              scope = 'usa', animation_frame= "Year",
              color_continuous_scale='YlGnBu',
              title = '<b>Geographic distribution of % votes by state</b>',
              height=600)