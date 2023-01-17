from bokeh.plotting import figure, output_file, show
from bokeh.io import output_notebook

import matplotlib.pyplot as plt
from scipy.stats import skew
import pandas as pd
import numpy as np
import seaborn as sns

output_notebook()

%matplotlib inline
import warnings
data = pd.ExcelFile('../input/zomato/zomato.xlsx').parse()
data.head()
data.shape
numerical_features   = data.select_dtypes(include = [np.number]).columns
categorical_features = data.select_dtypes(include= [np.object]).columns
numerical_features
categorical_features
col = ['Rating color', 'Rating text', 'Aggregate rating', 'Votes']
data[col].head(2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (20, 5))
sns.countplot('Rating color', data = data, ax = ax1)
sns.countplot('Rating text', data = data, hue = 'Rating color', ax= ax2,)
p = figure(plot_width=800, plot_height=400, title = "Votes Vs. Aggregate Voting")
p.xaxis.axis_label = 'Aggregate Rating'
p.yaxis.axis_label = 'Votes'

colormap = {'Excellent': 'green', 'Very Good': 'blue', 'Good': 'orange', 'Average': 'yellow', 
            'Not rated': 'black', 'Poor': 'red'}
colors = [colormap[x] for x in data['Rating text']]

p.asterisk(x = data['Aggregate rating'], y = data['Votes'], size=20, color=colors, alpha=0.7,)

show(p)
zero_rated_resturants = data[data['Aggregate rating'] == 0]
rated_resturants = data[data['Aggregate rating'] > 0]
fig, ax1 = plt.subplots(1, 1, figsize = (20, 5))
ax1.set_xlabel('Country Code')
ax1.set_ylabel('No. of resturants')
ax1.set_yscale('symlog')
current_palette_4 = sns.color_palette("Greys_r", 4)
sns.set_palette(current_palette_4)
sns.countplot('Country Code', data = zero_rated_resturants, ax = ax1).set_title("Countries having no rated resturants")
fig, ax = plt.subplots(2, 2, figsize = (20, 9), sharey=False, sharex=False)
ax[0][0].set_yscale('symlog')
palett = sns.color_palette("Blues_r")
sns.set_palette(palette=palett)
sns.countplot('City', data = zero_rated_resturants[zero_rated_resturants['Country Code']==1], ax= ax[0][0],).set_title("Country code 1")
sns.countplot('City', data = zero_rated_resturants[zero_rated_resturants['Country Code']==30], ax= ax[0][1]).set_title("Country code 30")
sns.countplot('City', data = zero_rated_resturants[zero_rated_resturants['Country Code']==215], ax= ax[1][0]).set_title("Country code 215")
sns.countplot('City', data = zero_rated_resturants[zero_rated_resturants['Country Code']==216], ax= ax[1][1]).set_title("Country code 216")
zero_rated_resturants[zero_rated_resturants['Country Code']== 1].head(30)
rated_resturants[rated_resturants['Country Code']== 1].head(20)
print(zero_rated_resturants['Has Table booking'].unique())
print(zero_rated_resturants['Has Online delivery'].unique())
print(zero_rated_resturants['Is delivering now'].unique())
print(zero_rated_resturants['Switch to order menu'].unique())
print(zero_rated_resturants['Price range'].unique())
# Reference table
country_code = pd.ExcelFile('../input/zomato/Country-Code.xlsx').parse()
country_code
from IPython.display import IFrame
IFrame('https://public.tableau.com/profile/rageeni#!/vizhome/ZomatoResturantsAnalysis/ZomatoResturantAnalysis?publish=yes', 
       width=1100, height=1000)