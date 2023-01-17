import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))
import statsmodels.formula.api as stats
from statsmodels.formula.api import ols
import sklearn
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error
import plotly.plotly as py 
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
happiness_2015 = pd.read_csv("../input/2015.csv")
happiness_2015.columns = ['Country', 'Region', 'Happiness_Rank', 'Happiness_Score',
       'Standard Error', 'Economy', 'Family',
       'Health', 'Freedom', 'Trust',
       'Generosity', 'Dystopia_Residual']
columns_2015 = ['Region', 'Standard Error']
new_dropped_2015 = happiness_2015.drop(columns_2015, axis=1)
happiness_2016 =  pd.read_csv("../input/2016.csv")
columns_2016 = ['Region', 'Lower Confidence Interval','Upper Confidence Interval' ]
dropped_2016 = happiness_2016.drop(columns_2016, axis=1)
dropped_2016.columns = ['Country', 'Happiness_Rank', 'Happiness_Score','Economy', 'Family',
       'Health', 'Freedom', 'Trust',
       'Generosity', 'Dystopia_Residual']
happiness_2017 =  pd.read_csv("../input/2017.csv")
columns_2017 = ['Whisker.high','Whisker.low' ]
dropped_2017 = happiness_2017.drop(columns_2017, axis=1)
dropped_2017.columns = ['Country', 'Happiness_Rank', 'Happiness_Score','Economy', 'Family',
       'Health', 'Freedom', 'Trust',
       'Generosity', 'Dystopia_Residual']
frames = [new_dropped_2015, dropped_2016, dropped_2017]
happiness = pd.concat(frames)
happiness.head()
happiness.describe()
data6 = dict(type = 'choropleth', 
           locations = happiness['Country'],
           locationmode = 'country names',
           z = happiness['Happiness_Rank'], 
           text = happiness['Country'],
          colorscale = 'Viridis', reversescale = False)
layout = dict(title = 'Happiness Rank Across the World', 
             geo = dict(showframe = False, 
                       projection = {'type': 'Mercator'}))
choromap6 = go.Figure(data = [data6], layout=layout)
iplot(choromap6)
data2 = dict(type = 'choropleth', 
           locations = happiness['Country'],
           locationmode = 'country names',
           z = happiness['Happiness_Score'], 
           text = happiness['Country'],
           colorbar = {'title':'Happiness'})
layout = dict(title = 'Happiness Score Across the World', 
             geo = dict(showframe = False, 
                       projection = {'type': 'Mercator'}))
choromap3 = go.Figure(data = [data2], layout=layout)
iplot(choromap3)
trace4 = go.Scatter(
    x = happiness.Happiness_Score,
    y = happiness.Happiness_Rank,
    mode = 'markers'
)
data4 = [trace4]
layout = go.Layout(
    title='Happiness Rank Determined by Score',
    xaxis=dict(
        title='Happiness Score',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Happiness Rank',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig4 = go.Figure(data=data4, layout=layout)
iplot(fig4)
drop_rank = happiness.drop("Happiness_Rank", axis = 1)
corr_matrix_happy = drop_rank.corr()
trace_corr_happy = go.Heatmap(z=np.array(corr_matrix_happy), x=corr_matrix_happy.columns, y=corr_matrix_happy.columns)
data_happy=[trace_corr_happy]
iplot(data_happy)
dropped_happy = happiness.drop(["Country", "Happiness_Rank"], axis=1)
dropped_happy.head()
#http://bigdata-madesimple.com/how-to-run-linear-regression-in-python-scikit-learn/
from sklearn.linear_model import LinearRegression
X = dropped_happy.drop("Happiness_Score", axis = 1)
lm = LinearRegression()
lm.fit(X, dropped_happy.Happiness_Score)
print("Estimated Intercept is", lm.intercept_)
print("The number of coefficients in this model are", lm.coef_)
coef = zip(X.columns, lm.coef_)
coef_df = pd.DataFrame(list(zip(X.columns, lm.coef_)), columns=['features', 'coefficients'])
coef_df
lm.predict(X)[0:100]
trace = go.Scatter(
    x = lm.predict(X),
    y = dropped_happy.Happiness_Score,
    mode = 'lines+markers'
)
data = [trace]
layout = go.Layout(
    title='Happiness Score vs. Predicted Happiness Score',
    xaxis=dict(
        title='Happiness Score',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    ),
    yaxis=dict(
        title='Predicted Happiness Score',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='#7f7f7f'
        )
    )
)

fig = go.Figure(data=data, layout=layout)
iplot(fig)

msehappy = np.mean((dropped_happy.Happiness_Score - lm.predict(X)) ** 2 ) 
print(msehappy)
lm2=LinearRegression()
lm2.fit(X[['Family']], dropped_happy.Happiness_Score)
msefamily = np.mean((dropped_happy.Happiness_Score - lm2.predict(X[['Family']])) **2)
print(msefamily)