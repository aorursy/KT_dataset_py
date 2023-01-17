import numpy as np # linear algebra

import pandas as pd 

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from plotly import tools

import plotly

plotly.offline.init_notebook_mode(connected=True) #to plot graph with offline mode in Jupyter notebook

import plotly.graph_objs as go



from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
# Read CSV

df = pd.read_csv("../input/winemag-data-130k-v2.csv")



#remove unnecessary column

df_cleaned = df.drop('Unnamed: 0', 1)

pd.set_option('display.max_columns', 500)

print(df_cleaned.shape)

#df_cleaned.info()

df_cleaned.head()
###########################

#Analyse based on Country#

#########################

countries = df_cleaned['country'].value_counts().sort_values(ascending=True)

trace_countries = [go.Pie(

                   labels = countries.index,

                   values = countries.values)]



trace_countries_layout = go.Layout(

                            title = 'Market Share of Wine By Countries')



trace_countries_fig = go.Figure(data = trace_countries, layout = trace_countries_layout)





plotly.offline.iplot(trace_countries_fig, filename = 'country_produce_wine')
#####################

#Bivariate Analysis#

###################

print(df_cleaned[['points','price']].describe())



points_price_col = ['points','price']

df_cleaned[points_price_col].plot(kind='box',subplots=True, title = 'Boxplots on Wine Points & Price')



##############################

#Distribution of Wine Rating#

############################

hist_points = [go.Histogram(

                x = df_cleaned['points']

)]



hist_points_layout = go.Layout(

                        title = 'Distribution of Wine Points/Rating',

                        xaxis = dict(title = 'Points'),

                        yaxis = dict(title = 'Number of rating'))



hist_points_fig = go.Figure(data = hist_points, layout = hist_points_layout)

    

plotly.offline.iplot(hist_points_fig, filename = 'distribution_wine_points')



#############################

#Distribution of Wine Price#

###########################

hist_price = [go.Histogram(

                x = df_cleaned['price'])]



hist_price_layout = go.Layout(

                        title = 'Distribution of Wine Price',

                        xaxis = dict(title = 'Price'),

                        yaxis = dict(title = 'Frequency'))





hist_price_fig = go.Figure(data = hist_price, layout = hist_price_layout)



plotly.offline.iplot(hist_price_fig, filename='distribution_wine_price')
#Impute NaN field in price with its median price

df_cleaned[['price']] = df_cleaned[['price']].fillna(value=25)

df_cleaned.corr()
#split data into x-array and y-array

X_var = df_cleaned[['points']]

y_var = df_cleaned['price']
df_cleaned.plot(x='points',y='price',style='o')