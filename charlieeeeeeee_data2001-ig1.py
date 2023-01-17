import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

relevant_cols = pd.read_csv("../input/cv19_regression.csv") # pulled from SQL and preprocessed


fig = make_subplots(rows=1, cols=2)


reg1 = LinearRegression().fit(np.vstack(relevant_cols['vulnerability_score_4']), relevant_cols['num_of_tests'])
relevant_cols['bestfit4_tests'] = reg1.predict(np.vstack(relevant_cols['vulnerability_score_4']))

reg2 = LinearRegression().fit(np.vstack(relevant_cols['vulnerability_score_5']), relevant_cols['num_of_tests'])
relevant_cols['bestfit5_tests'] = reg2.predict(np.vstack(relevant_cols['vulnerability_score_5']))


fig.add_trace(go.Scatter(x=relevant_cols['vulnerability_score_4'], y=relevant_cols['num_of_tests'], 
                         name = "vulnerability_score_4",
                         text=relevant_cols['area_name'],
                         mode='markers', marker=dict(
                             color=relevant_cols['num_of_tests'],
                             colorscale='rainbow')
                         ),
              row=1, col=1)


fig.add_trace(go.Scatter(name='line of best fit', x=relevant_cols['vulnerability_score_4'],
                         y=relevant_cols['bestfit4_tests'], mode='lines'),
             row=1, col=1)


fig.add_trace(go.Scatter(x=relevant_cols['vulnerability_score_5'], y=relevant_cols['num_of_tests'], 
                         name = "vulnerability_score_5",
                         text=relevant_cols['area_name'],
                         mode='markers',marker=dict(
                             color=relevant_cols['num_of_tests'],
                             colorscale='rainbow')), 
              row=1, col=2)

fig.add_trace(go.Scatter(name='line of best fit', x=relevant_cols['vulnerability_score_5'],
                         y=relevant_cols['bestfit5_tests'], mode='lines'),
             row=1, col=2)

fig.update_layout(
    xaxis_title="vulnerability_score",
    yaxis_title="number_of_cases",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)

# fig = go.Figure(data=go.Scatter(x=data['Postal'],
#                                 y=data['Population'],
#                                 mode='markers',
#                                 marker_color=data['Population'],
#                                 text=data['State'])) # hover text goes here


fig.show()

fig = make_subplots(rows=1, cols=2)

reg3 = LinearRegression().fit(np.vstack(relevant_cols['vulnerability_score_4']), relevant_cols['num_of_cases'])
relevant_cols['bestfit4_cases'] = reg1.predict(np.vstack(relevant_cols['vulnerability_score_4']))

reg4 = LinearRegression().fit(np.vstack(relevant_cols['vulnerability_score_5']), relevant_cols['num_of_cases'])
relevant_cols['bestfit5_cases'] = reg2.predict(np.vstack(relevant_cols['vulnerability_score_5']))



fig.add_trace(go.Scatter(x=relevant_cols['vulnerability_score_4'], y=relevant_cols['num_of_cases'],
                         name = "vulnerability_score_4",
                         text=relevant_cols['area_name'],
                         mode='markers', marker=dict(
                             color=relevant_cols['num_of_tests'],
                             colorscale='rainbow')),
              row=1, col=1)

fig.add_trace(go.Scatter(name='line of best fit', x=relevant_cols['vulnerability_score_4'],
                         y=relevant_cols['bestfit4_cases'], mode='lines'),
             row=1, col=1)

fig.add_trace(go.Scatter(x=relevant_cols['vulnerability_score_5'], y=relevant_cols['num_of_cases'], 
                         name = "vulnerability_score_5",
                         text=relevant_cols['area_name'],
                         mode='markers',marker=dict(
                             color=relevant_cols['num_of_tests'],
                             colorscale='rainbow')),
              row=1, col=2)

fig.add_trace(go.Scatter(name='line of best fit', x=relevant_cols['vulnerability_score_5'],
                         y=relevant_cols['bestfit5_cases'], mode='lines'),
             row=1, col=2)

fig.update_layout(
    xaxis_title="vulnerability_score",
    yaxis_title="number_of_tests",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)

fig.show()