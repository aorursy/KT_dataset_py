import plotly.offline as ply
import plotly.graph_objs as go
ply.init_notebook_mode(connected=True)
import pandas as pd
import numpy as np

df=pd.read_csv('../input/kc_house_data.csv')

df.head()
trace1 = go.Scattergl(
    x=df.sqft_living15,
    y=df.price,
    mode='markers',
    marker=dict(
        opacity=0.5
    ),
    #showlegend=True
)
data=[trace1]

layout = go.Layout(
    title='Price vs. Living Room Area',
    xaxis=dict(
        title='Living room area (sq. ft.)'
    ),
    yaxis=dict(
        title='Price ($)'
    ),
    hovermode='closest',
    #showlegend=True
)

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
dataPoints = go.Scattergl(
    x=df.sqft_living15,
    y=np.log(df.price),
    mode='markers',
    marker=dict(
        opacity=0.25,
        line=dict(
            color='white'
        )
    ),
    name='Data points'
)

data=[dataPoints]

layout.update(
    yaxis=dict(
        title='Log(Price)'
    )
)

figure.update(
    data=data,
    layout=layout
)
ply.iplot(figure)
import seaborn as sns

sns.regplot(df.sqft_living15, np.log(df.price))
m,b = np.polyfit(df.sqft_living15, np.log(df.price), 1)
bestfit_y = (df.sqft_living15 * m + b)
lineOfBestFit=go.Scattergl(
    x=df.sqft_living15,
    y=bestfit_y,
    name='Line of best fit',
    line=dict(
        color='red',
    )
)

data=[dataPoints, lineOfBestFit]
figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
dataPoints = go.Scattergl(
    x=df.sqft_living15,
    y=np.log(df.price),
    mode='markers',
    text=[f'Living Room Area:{df.at[i, "sqft_living15"]} sq.ft.<br>Grade:{df.at[i, "grade"]}<br>Price:${df.at[i, "price"]}' for i in range(len(df))],
    marker=dict(
        opacity=0.75,
        color=df.grade,
        showscale=True,
        colorscale='Jet',
        colorbar=dict(
            title='Grade'
        ),
    ),
    name='Data points'
)

data=[dataPoints]

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)
grades=sorted(df.grade.unique())

data=[]
for g in grades:
    df_grade=df[df.grade==g]
    data.append(
        go.Scattergl(
            x=df_grade.sqft_living15,
            y=np.log(df_grade.price),
            mode='markers',
            text=[f'Living Room Area:{df_grade.at[i, "sqft_living15"]} sq.ft.<br>Grade:{df_grade.at[i, "grade"]}<br>Price:${df_grade.at[i, "price"]}' for i in df_grade.index],
            marker=dict(
                opacity=0.75,
            ),
            name='Grade:'+str(g)
        )
    )

figure = go.Figure(data=data, layout=layout)

ply.iplot(figure)    
subplots=['No. of bedrooms', 'No. of bathrooms', 'Condition', 'Grade', 'Waterfront']
subplot_cols=['bedrooms', 'bathrooms', 'condition', 'grade', 'waterfront']

from plotly.tools import make_subplots
figure=make_subplots(rows=5, cols=1, subplot_titles=['Breakup by '+col for col in subplots])

for i in range(len(subplots)):
    col_name=subplots[i]
    col=subplot_cols[i]
    col_values=sorted(df[col].unique())
    for value in col_values:
        df_subset=df[df[col]==value]
        trace=go.Scattergl(
            x=df_subset.sqft_living15,
            y=np.log(df_subset.price),
            mode='markers',
            text=[f'Living Room Area:{df_subset.at[i, "sqft_living15"]} sq.ft.<br>{col_name}:{df_subset.at[i, col]}<br>Price:${df_subset.at[i, "price"]}' for i in df_subset.index],
            marker=dict(
                opacity=0.75,
            ),
            name=col_name+':'+str(value),
            showlegend=False
        )
        figure.append_trace(trace, i+1, 1)

figure['layout'].update(
    height=2000, 
    title='Price vs. Living room area - subplots', 
    hovermode='closest',
    xaxis=dict(title='Living room area (sq. ft.)'),
    xaxis2=dict(title='Living room area (sq. ft.)'),
    xaxis3=dict(title='Living room area (sq. ft.)'),
    xaxis4=dict(title='Living room area (sq. ft.)'),
    xaxis5=dict(title='Living room area (sq. ft.)'),
    yaxis=dict(title='Log(Price)'),
    yaxis2=dict(title='Log(Price)'),
    yaxis3=dict(title='Log(Price)'),
    yaxis4=dict(title='Log(Price)'),
    yaxis5=dict(title='Log(Price)'),
)

ply.iplot(figure)