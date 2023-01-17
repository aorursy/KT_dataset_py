# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
from matplotlib import pyplot as plt
import seaborn as sns
import plotly 
#Plotly's Python graphing library makes interactive, publication-quality graphs online.
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import cufflinks as cf
#Cufflinks binds Plotly directly to pandas dataframes.
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv("../input/googleplaystore.csv")
print('Data overview')
print('Delete duplicate values to calculate the total number of Apps')
df.drop_duplicates(subset='App',inplace=True)
df=df[df['Android Ver']!=np.nan]
df=df[df['Android Ver']!='NaN']
df=df[df['Installs']!='Free']
df = df[df['Installs'] != 'Paid']
print('Number of apps in the dataset : ' , len(df))
print(df.sample(10))
print('Data cleaning')
print('Encode undefined values as missing values')
print('Convert the unit of measure of size into "M" ')
df['Size']=df['Size'].fillna(0)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)
df['Size'] = df['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)
df['Size']=df['Size'].apply(lambda x:str(x).replace(',','')if 'M' in str(x) else x)
df['Size']=df['Size'].apply(lambda x:float(str(x).replace('k',''))/1000 if 'k'in str(x) else x)
df['Size']=df['Size'].apply(lambda x:float(x))
print('Convert the values of Installs,Price to float') 
df['Installs'] = df['Installs'].apply(lambda x: x.replace('+', '') if '+' in str(x) else x)
df['Installs'] = df['Installs'].apply(lambda x: x.replace(',', '') if ',' in str(x) else x)
df['Installs']=df['Installs'].apply(lambda x:float(x))
df['Price'] = df['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
df['Price'] = df['Price'].apply(lambda x: float(x))
print('Convert the values of Reviews to int')
df['Reviews'] = df['Reviews'].apply(lambda x: int(x))
print('Convert Category values to lowercase')
df['Category'] = df['Category'].apply(lambda x:str(x).lower())
print(df.sample(10))
print('Basic EDA')
from sklearn import preprocessing as pre
rating=df['Rating'].dropna()
size=df['Size'].dropna()
price = df['Price'].dropna()
#Normalized processing
install=df['Installs'].dropna()
install_s=pre.scale(install)
review=df['Reviews'].dropna()
review_s=pre.scale(review)

#Convert to factor type
# 0 - Free, 1 - Paid
df['Type'] = pd.factorize(df['Type'])[0]
type=df['Type'].dropna()
#pairplot
sns.pairplot(pd.DataFrame(list(zip(rating,size,install_s,review_s,type,price)),columns=['rating','size','install','review','type','price']),
            hue='type',palette="husl",kind='scatter')
print('Calculate the most popular apps in the market')
app_num=df['Category'].value_counts().sort_values(ascending=True)
app_num.head(3)
trace=go.Pie(labels=app_num.index,values=app_num.values,hoverinfo='label+percent')
plotly.offline.iplot([trace])
print('Family and Game apps are the most popular')
print('average rating of apps')
data = [go.Histogram(
        x = df.Rating,
        xbins = {'start': 1, 'size': 0.1, 'end' :5})]
print(np.mean(df['Rating']))
plotly.offline.iplot(data)
print('Most APP ratings are concentrated around 4.2 points.')
print('About category:')
print('Is there a significant difference in Rating between different categories?')
groups = df.groupby('Category').mean()
groups
groups = df.groupby('Category').filter(lambda x: len(x) > 286).reset_index()
array = groups['Rating'].hist(by=groups['Category'], sharex=True, figsize=(20,20))
#One Way Anova Test
import scipy.stats as stats
f=stats.f_oneway(df.loc[df.Category=='family']['Rating'].dropna(),
                df.loc[df.Category=='game']['Rating'].dropna(),
                df.loc[df.Category=='business']['Rating'].dropna(),
                df.loc[df.Category=='tools']['Rating'].dropna(),
                df.loc[df.Category=='medical']['Rating'].dropna())
print(f)
print('The p-value is extremely small, hence,there is a significant difference in the mean of the ratings of various APPs.')
groups=df.groupby('Category').filter(lambda x:len(x)>=170).reset_index()

data = [{
    'y': df.loc[df.Category==category]['Rating'], 
    'type':'violin',
    'name' : category,
    'showlegend':False,
    #'marker': {'color': 'Set2'},
    } for i,category in enumerate(list(set(groups.Category)))]
layout = {'title' : 'App ratings across major categories',
        'xaxis': {'tickangle':-40},
        'yaxis': {'title': 'Rating'},
          'plot_bgcolor': 'rgb(250,250,250)',
          'shapes': [{
              'type' :'line',
              'x0': -.5,
              'y0': np.nanmean(list(groups.Rating)),
              'x1': 19,
              'y1': np.nanmean(list(groups.Rating)),
              'line': { 'dash': 'dashdot'}
          }]
          }
plotly.offline.iplot({'data': data, 'layout': layout})
print('The ratings of all kinds of APPs are generally concentrated on 4 points and above, and the overall evaluation is good.')
print('In comparison, Health and Fitness and Books and Reference produces the highest quality Apps with 50% apps having a rating greater than 4.5.')
print('But there are also some garbage Apps, such as lifestyle, family, and finance.')
print('About size:')
print('Does the size of APPs affect the rating?')
groups=df.groupby('Category').filter(lambda x: len(x) >= 50).reset_index()
sns.set_style('darkgrid')
ax=sns.jointplot(df['Size'],df['Rating'],kind='hex',color='mediumaquamarine')
sns.set_style('darkgrid')
ax=sns.jointplot(df['Size'],df['Rating'],kind='kde',color='mediumaquamarine')
print('The size of Apps is generally distributed within 60M. Among them, the most concentrated within 20M, and the score is relatively higher.')
ax=sns.jointplot(df['Size'],df['Rating'],kind='reg',color='olive')
print('Regression analysis shows that the P value is much smaller than 0, and there is a significant correlation between rating and size.')
print('About price:')
print('How do App prices impact App rating?')
paid_apps = df[df.Price>0]
p = sns.jointplot( "Price", "Rating", paid_apps,color='crimson',
                 marginal_kws=dict(bins=15, rug=True),
                 annot_kws=dict(stat="r"),
                 kind='scatter')

print('Exploring which Apps are so expensive and their ratings')
df[['Category','App','Price','Rating']][df.Price>300]
print('Surprisingly, these expensive Apps don’t get a very high rating.')
print('The vast majority of Apps are priced between $0 and $20, and only a handful of Apps cost more than $400.')
print('Correlation analysis shows that there is a certain negative relationship between price and Apps rating.')
subset_df = df[df.Category.isin(['game', 'family', 'tools', 'medical', 'finance', 'business'])]
subset_df_price = subset_df[subset_df.Price<100]
sns.set_style('whitegrid')

fig,ax = plt.subplots()
title = ax.set_title('App pricing trend across categories')
sns.stripplot(x="Price", y="Category",data=subset_df_price,jitter=True)
print('Games and tools Apps are relatively affordable, while medical Apps are relatively expensive.')
print('Distribution of paid and free Apps across categories.')
print('Based on the above analysis results, explore the top five App payment.')
print('Correlation analysis')
corrmat = df.corr()
f, ax = plt.subplots()
p =sns.heatmap(corrmat, annot=True, cmap="YlGnBu")
#import pyecharts
#from pyecharts import Polar
#groups = df.groupby('Category').mean()
#print(groups.head())
#print('Explore the top five app reviews')
#import pyecharts
#radius =['art&design','auto&vehicles','beauty','books&reference','business']
#polar =Polar("Reviews in Apps", width=1200, height=600)
#polar.add("Number of reviews", [22175,13690,7476,75321,23548], radius_data=radius, type='barRadius', is_stack=True)
#polar.show_config()

print('The books&reference Apps have the largest number of comments, so it shows that it receives a lot of attention.')
print('The beauty Apps have received less reviews, so future research can explore the reasons deeply.')
print("Correlation between 'Installs' and 'Reviews':",df['Installs'].corr(df['Reviews']))
print("Correlation analysis verifies that 'Reviews' and 'Installs' have strong correlations")
df_copy = df.copy()
df_copy = df_copy[df_copy.Reviews > 0]
df_copy = df_copy[df_copy.Installs > 0]
df_copy['Installs']=np.log10(df['Installs'])
df_copy['Reviews']=np.log10(df['Reviews'])

sns.lmplot('Reviews','Installs',data=df_copy)
ax=plt.gca()
ax.set_title('Reviews and Installs')