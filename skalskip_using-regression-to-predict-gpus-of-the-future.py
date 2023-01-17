import numpy as np

import pandas as pd

from scipy.optimize import curve_fit

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
# Let's import dataset from csv file

dataset = pd.read_csv('../input/All_GPUs.csv')
dataset.head(10)
dataset.info()
import missingno as msno

# Nullity or missing values by columns

#msno.matrix(df=dataset, figsize=(20, 8), color=(255/255, 83/255, 51/255))

msno.bar(df=dataset, figsize=(20, 8), color=(255/255, 83/255, 51/255))
key_columns = ['Best_Resolution', 'Core_Speed', 'Manufacturer', 'Memory', 'Memory_Bandwidth', 'Name', 'Release_Date']

dataset = dataset[key_columns]
dataset['Release_Date']=dataset['Release_Date'].str[1:-1]

dataset=dataset[dataset['Release_Date'].str.len()==11]

dataset['Release_Date']=pd.to_datetime(dataset['Release_Date'], format='%d-%b-%Y')

dataset['Release_Year']=dataset['Release_Date'].dt.year

dataset['Release_Month']=dataset['Release_Date'].dt.month

dataset['Release']=dataset['Release_Year'] + dataset['Release_Month']/12
dataset['Memory'] = dataset['Memory'].str[:-3].fillna(0).astype(int)
def countPixels(x):

    if pd.isnull(x):

        return 800*600;

    else:

        values = x.split(' x ')

        return(int(values[0]) * int(values[1]))



dataset['PixelNum'] = dataset['Best_Resolution'].apply(lambda x: countPixels(x))
plt.figure(figsize=(16,8))

sns.set_style("whitegrid")

sns.countplot(x="Release_Year", data=dataset, palette="hls");

plt.title('Grouping GPUs by Release Year', fontsize=20, fontweight='bold', y=1.05,)

plt.xlabel('Number GPUs', fontsize=15)

plt.ylabel('Release Year', fontsize=15)

plt.show()
data = []



for manufacturer in dataset['Manufacturer'].unique():

    

    trace_dataset = dataset.loc[dataset['Manufacturer'] == manufacturer]

    

    trace = go.Scatter(

        x = trace_dataset['Release_Year'],

        y = trace_dataset['Memory'],

        mode = 'markers',

        name = manufacturer,

        marker= dict(symbol="circle",

                    size = trace_dataset['PixelNum']/100000,

                    opacity = 0.4,

                    line= dict(width=1,

                               color = 'rgb(255, 255, 255)'

                              ),

        ),

        text= dataset['Name']

    )

    

    data.append(trace)

    

layout = dict(title = 'GPU Memory vs Year of Release by Manufacturer',

              width=1000,

              height=800,

              paper_bgcolor='rgb(243, 243, 243)',

              plot_bgcolor='rgb(243, 243, 243)',

              yaxis = dict(title= 'GPUs Memory',

                           ticklen= 5,

                           gridcolor='rgb(255, 255, 255)',

                           gridwidth= 2),

              xaxis = dict(title= 'Year of Release',

                           ticklen= 5,

                           gridcolor='rgb(255, 255, 255)',

                           gridwidth= 2)

             )



fig= go.Figure(data=data, layout=layout)

py.iplot(fig)
# Transformed dataset - counting how many GPUs were made by each manufacturer each year

market_share = dataset.reset_index().groupby(['Release_Year','Manufacturer'])["index"].count().reset_index(name="count")

# Transformed dataset - counting how many GPUs were made each year

market_share['Sum_By_Year']  = market_share['count'].groupby(market_share["Release_Year"]).transform('sum')

# List that will hold our data dictionaries

data = []



markers = [

    dict(

        color='rgba(55, 128, 191, 0.7)',

        line=dict(

            color='rgba(55, 128, 191, 1.0)',

            width=2,

        )

    ),

    dict(

        color='rgba(219, 64, 82, 0.7)',

        line=dict(

            color='rgba(219, 64, 82, 1.0)',

            width=2,

        )

    ),

    dict(

        color='rgba(0, 168, 107, 0.7)',

        line=dict(

            color='rgba(0, 168, 107, 1.0)',

            width=2,

        )

    ),

    dict(

        color='rgba(250, 92, 0, 0.7)',

        line=dict(

            color='rgba(250, 92, 0, 1.0)',

            width=2,

        )

    )

]



for i, manufacturer in enumerate(dataset['Manufacturer'].unique()):

    

    trace_dataset = market_share.loc[market_share['Manufacturer'] == manufacturer]

    

    

    trace = go.Bar(

        x = trace_dataset['Release_Year'],

        y = round(trace_dataset['count'] / trace_dataset['Sum_By_Year'] * 100, 2),

        name= manufacturer,

        marker = markers[i],

    )

    

    data.append(trace)

    

layout = go.Layout(

    barmode='stack',

    title = 'GPU manufacturers market share',

    width=1000,

    height=500,

    paper_bgcolor='rgb(243, 243, 243)',

    plot_bgcolor='rgb(243, 243, 243)',

    yaxis = dict(title= 'Manufacturers market share [%]'),

    xaxis = dict(title= 'Manufacturers')

)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
plt.figure(figsize=(16,10))

sns.set_style("whitegrid")

plt.title('GPU Memory vs Year of Release', fontsize=20, fontweight='bold', y=1.05,)

plt.xlabel('Year of Release', fontsize=15)

plt.ylabel('GPU Memory', fontsize=15)



years = dataset["Release"].values

memory = dataset["Memory"].values



plt.scatter(years, memory, edgecolors='black')

plt.show()
# Numpy array that holds unique release year values

year_arr = dataset.sort_values("Release_Year")['Release_Year'].unique()

# Numpy array that holds mean values of GPUs memory for each year

memory_arr_mean = dataset.groupby('Release_Year')['Memory'].mean().values

# Numpy array that holds median values of GPUs memory for each year

memory_arr_median = dataset.groupby('Release_Year')['Memory'].median().values



# Minimal value of release year from dataset

year_min = year_arr[0]

# Median size of memory in year_min

memory_min = memory_arr_median[0]
plt.figure()

plt.figure(figsize=(16,8))

plt.title('GPU Memory vs Year of Release', fontsize=20, fontweight='bold', y=1.05,)

plt.xlabel('Year of Release', fontsize=15)

plt.ylabel('GPU Memory', fontsize=15)

sns.set_style("whitegrid")

plt.plot(year_arr, memory_arr_mean, label="Mean")

plt.plot(year_arr, memory_arr_median, label="Median")

plt.legend(loc=4, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")

plt.show()
# Short function to calculate size of memory based on Moore's law

def calculateMooresValue(x, y_trans):

    return memory_arr_median[0] * 2**((x-y_trans)/2)



# GPU Memory Size calculation based on Moore's Law

y_pred_moore_law_teoretic = calculateMooresValue(year_arr, int(year_min))
# Fitting exponential curve to dataset

def exponentialCurve(x, a, b, c):

    return a*2**((x-c)*b)



popt, pcov = curve_fit(exponentialCurve,  year_arr, memory_arr_mean,  p0=(2, 0.5, 1998))

y_pred_moore_law_fitted = exponentialCurve(year_arr, *popt)
plt.figure()

plt.figure(figsize=(16,8))

plt.title('GPU Memory vs Year of Release [Logaritmic scale]', fontsize=20, fontweight='bold', y=1.05,)

plt.xlabel('Year of Release', fontsize=15)

plt.ylabel('GPU Memory', fontsize=15)

sns.set_style("whitegrid")

plt.loglog(year_arr, memory_arr_mean, label="Mean", basex= 2)

plt.loglog(year_arr, memory_arr_median, label="Median", basex= 2)

plt.loglog(year_arr, y_pred_moore_law_teoretic, label="Moore's law teoretic", basex= 2)

plt.loglog(year_arr, y_pred_moore_law_fitted, label="Moore's law fitted", basex= 2)

plt.legend(loc=4, prop={'size': 15}, frameon=True,shadow=True, facecolor="white", edgecolor="black")

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures



# Fitting Polynomial Regression to the dataset

poly_reg_2 = PolynomialFeatures(degree = 2, include_bias=False)

poly_reg_3 = PolynomialFeatures(degree = 3, include_bias=False)



X_poly_2 = poly_reg_2.fit_transform(year_arr.reshape(-1, 1))

X_poly_3 = poly_reg_3.fit_transform(year_arr.reshape(-1, 1))



lin_reg_2 = LinearRegression()

lin_reg_3 = LinearRegression()



lin_reg_2.fit(X_poly_2, memory_arr_mean)

lin_reg_3.fit(X_poly_3, memory_arr_mean)



y_pred_lin_reg_2 = lin_reg_2.predict(poly_reg_2.fit_transform(year_arr.reshape(-1, 1)))

y_pred_lin_reg_3 = lin_reg_3.predict(poly_reg_3.fit_transform(year_arr.reshape(-1, 1)))
from sklearn.metrics import r2_score



# 2nd degree curve

score = r2_score(y_pred_lin_reg_2, memory_arr_mean)

print("r2 of 2nd degree curve is equal " + str(round(score, 3)))

# 3rd degree curve

score = r2_score(y_pred_lin_reg_3, memory_arr_mean)

print("r2 of 3rd degree curve is equal " + str(round(score, 3)))

# Teoretic Moore's Law curve

score = r2_score(y_pred_moore_law_teoretic, memory_arr_mean)

print("r2 of Teoretic Moore's Law curve is equal " + str(round(score, 3)))

# Fitted Moore's Law curve

score = r2_score(y_pred_moore_law_fitted, memory_arr_mean)

print("r2 of Fitted Moore's Law curve is equal " + str(round(score, 3)))
X_grid = np.arange(min(year_arr), max(year_arr) + 5, 0.1)

X_grid = X_grid.reshape((len(X_grid), 1))



y_pred_lin_reg_2 = lin_reg_2.predict(poly_reg_2.fit_transform(X_grid))

y_pred_lin_reg_3 = lin_reg_3.predict(poly_reg_3.fit_transform(X_grid))



X_grid = X_grid.flatten()



y_pred_moore_law_teoretic = calculateMooresValue(X_grid, int(year_min))

y_pred_moore_law_fitted = exponentialCurve(X_grid, *popt)



trace1 = go.Scatter(

    x = X_grid,

    y = y_pred_lin_reg_2,

    mode = 'lines',

    name = '2nd degree polynomial'

)

trace2 = go.Scatter(

    x = X_grid,

    y = y_pred_lin_reg_3,

    mode = 'lines',

    name = '3rd degree polynomial'

)

trace3 = go.Scatter(

    x = X_grid,

    y = y_pred_moore_law_teoretic,

    mode = 'lines',

    name = "Teoretic Moore's Law curve"

)

trace4 = go.Scatter(

    x = X_grid,

    y = y_pred_moore_law_fitted,

    mode = 'lines',

    name = "Fitted Moore's Law curve"

)



data = [trace1, trace2, trace3, trace4]



layout = go.Layout(

    title = 'Fitting regression model into dataset',

    width=1000,

    height=500,

    paper_bgcolor='rgb(243, 243, 243)',

    plot_bgcolor='rgb(243, 243, 243)',

    yaxis = dict(title= 'GPU Memory'),

    xaxis = dict(title= 'Year of Release'),

    legend=dict(

        x=0,

        y=1,

        traceorder='normal',

        font=dict(

            family='sans-serif',

            size=12,

            color='#000'

        ),

        bgcolor='#E2E2E2',

        bordercolor='#FFFFFF',

        borderwidth=2

    )

)

fig = go.Figure(data=data, layout=layout)



py.iplot(fig)
memory_2025 = exponentialCurve(2025, *popt)
print("Predicted mean size of GPU memory in 2025 is " + str(round(int(memory_2025) / 1024, 2)) + " GB.")