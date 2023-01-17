# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly as plotly

import seaborn as sns

from sklearn import preprocessing
from plotly import __version__

import plotly.offline as py 

from plotly.offline import init_notebook_mode, plot

init_notebook_mode(connected=True)

from plotly import tools

import plotly.graph_objs as go

import plotly.express as px

import folium

from folium.plugins import MarkerCluster

from folium import plugins



from sklearn.preprocessing import LabelEncoder,OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV

from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV, LinearRegression, ElasticNet,  HuberRegressor

from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR

from sklearn.kernel_ridge import KernelRidge

from sklearn.utils import resample



import statsmodels

import statsmodels.api as sm



from keras import models, layers, optimizers, regularizers

from keras.utils.vis_utils import model_to_dot

from keras.layers import Conv2D, MaxPooling2D



#importing Leaky ReLu to use as an activation function

from keras.layers import LeakyReLU



from IPython.display import SVG
df_bnb = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

df_bnb.head()

print("The amount of listings of Airbnbs are: ", len(df_bnb))
###Finding the number of null data points 

df_bnb.isnull().sum()
df_bnb.drop(['host_name','last_review','id'],axis = 1, inplace = True)

df_bnb.head()
#Replacing the N/A with 0 for the reviews

df_bnb.fillna({'reviews_per_month': 0},inplace=True)
#Seeing the datatypes

df_bnb.dtypes
##Filling in null values for name. 

df_bnb.fillna({'name': ""},inplace=True)
#Next we want to see what are the unique neighbourhoods

hoods = df_bnb.neighbourhood.unique()

print(hoods)

print(len(hoods))
#Describing the data

df_bnb.describe()
print("The shape is: ", df_bnb.shape)
df_bnb.info()
corr = df_bnb.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

plt.show()
##Seeing what kind of room types are unique 

print("The unique types of rooms are: ",df_bnb.room_type.unique())
#Next, to ease confusion and settle with vernicular, we will change the name of neighbourhood_group to borough
df_bnb.rename(columns = {'neighbourhood_group': 'borough'},inplace = True)
df_bnb.hist(edgecolor='blue',linewidth=1.0,figsize=(30,30))

plt.figure(figsize = (30,30))

sns.pairplot(df_bnb,height=4,diag_kind = 'hist')
plt.figure(figsize = (10,10))

sns.scatterplot(x='longitude',y='latitude',hue = 'neighbourhood',s=20,data=df_bnb)
plt.figure(figsize = (10,10))

sns.scatterplot(x='longitude',y='latitude',hue = 'borough',s=20,data=df_bnb)
import folium

from folium.plugins import HeatMap





m=folium.Map([40.7122,-74.0000],zoom_start=11)

HeatMap(df_bnb[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)

display(m)
data = df_bnb.neighbourhood.value_counts()[:10]

plt.figure(figsize=(12, 8))

x = list(data.index)

y = list(data.values)

x.reverse()

y.reverse()



plt.title("Most Popular Neighbourhood")

plt.ylabel("Neighbourhood Area")

plt.xlabel("Number of guest Who host in this Area")



plt.barh(x, y)
top_hoods = df_bnb.loc[df_bnb['neighbourhood'].isin(['Williamsburg','Bedford-Stuyvesant','Harlem','Bushwick',

                 'Upper West Side','Hell\'s Kitchen','East Village','Upper East Side','Crown Heights','Midtown','East Harlem', 'Greenpoint','Chelsea','Lower East Side','Astoria'])]



hoods_viz=sns.catplot( data=df_bnb.loc[df_bnb['neighbourhood'].isin(['Williamsburg','Bedford-Stuyvesant','Harlem','Bushwick',

                 'Upper West Side','Hell\'s Kitchen','East Village','Upper East Side','Crown Heights','Midtown','East Harlem', 'Greenpoint','Chelsea','Lower East Side','Astoria'])], x='neighbourhood', hue='borough', col='room_type', kind='count')

hoods_viz.set_xticklabels(rotation=90)
import wordcloud

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from PIL import Image



def makeWC(text, colormap="viridis", imageUrl=None):

    if imageUrl is not None:

        nyc = np.array(Image.open(imageUrl))

        wc = WordCloud(background_color="green", colormap=colormap, mask=nyc, contour_width=1.5, contour_color='steelblue')

    else:

        wc = WordCloud(background_color="green",width=1920, height=1080,max_font_size=200, max_words=200, colormap=colormap)

    wc.generate(text)

    

    # Show WordCloud

    f, ax = plt.subplots(figsize=(12, 12))

    plt.imshow(wc, interpolation="bilinear")

    plt.axis("off")

    plt.show()

    

##Create a data frame for names because wordcloud won't work with Null Values 

df_names = df_bnb['name']





names_list = " ".join([name for name in df_names])

makeWC(names_list, colormap="RdYlGn")
#room_type - price

res = df_bnb.groupby(["room_type"])['price'].aggregate(np.median).reset_index().sort_values('price')

sns.barplot(x='room_type', y="price", data=df_bnb, order=res['room_type'])
print(f"Average of price / night : ${df_bnb.price.mean():.2f}")

print(f"Maximum price / night : ${df_bnb.price.max()}")

print(f"Minimum price / night : ${df_bnb.price.min()}")
##Price Distribution Plot 

plt.figure(figsize=(5,5))

sns.distplot(df_bnb['price'])

plt.title('Price Distribution Plot',weight='bold')
plt.figure(figsize=(8,8))

sns.distplot(np.log(df_bnb.price+1))

plt.title("Logarithmic Price Distribution Plot",size=15, weight='bold')
from scipy import stats



plt.figure(figsize=(7,7))

stats.probplot(np.log(df_bnb.price+1), plot=plt)

plt.show()
#Here we see an interesting pattern where there are listings that are $0. 

df_bnb[df_bnb.price == 0]

##Further data cleaning to drop listings with prices >$1000/night as this is skewing our data.

print(df_bnb[df_bnb['price']>1000])

##These are typically either more extravegent listings or errors 

# We will drop the data above 500

df_bnb=df_bnb[df_bnb["price"]<500]
##Getting the price distribution 

price_norm = df_bnb[df_bnb.price < 600]

vio_price = sns.violinplot(data=price_norm,x='borough',y='price')

vio_price.set_title('Density Distribution of prices for Boroughs')
##Price Distributions



df_bnb.groupby('borough')['price'].describe()
#Host Id has no context

df_bnb = df_bnb.drop(['host_id'],axis=1)

#Feature Engineering



df_bnb['avail'] = df_bnb['availability_365']>353

df_bnb['low_avail'] = df_bnb['availability_365']< 12

df_bnb['no_reviews'] = df_bnb['reviews_per_month']==0
df_fin = df_bnb.loc[:,df_bnb.columns != 'name']



print(df_fin.dtypes)
#Encoding categorical features

cat = df_fin.select_dtypes(include=['object'])

print('Categorical features: {}'.format(cat.shape))

cat_code = pd.get_dummies(cat)
cat_code.head()
nums =  df_bnb.select_dtypes(exclude=['object'])

y = nums.price 

nums = nums.drop(['price'],axis = 1)



print('Numerical features: {}'.format(nums.shape))
X = np.concatenate((nums,cat_code),axis=1)

#X_df = pd.concat([nums,cat_code],axis=1)
##Shuffling the data

from sklearn.utils import shuffle





X,y = shuffle(X,y,random_state=0)

print(X)

print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=42)
##Random Forest Method

from sklearn.model_selection import RandomizedSearchCV

import numpy as np



n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]

max_features = ['sqrt','auto']

max_depth = [int(x) for x in np.linspace(10, 110, num = 6)]

max_depth.append(None)

min_samples_split = [2, 5, 10]

min_samples_leaf = [1, 2, 4]

bootstrap = [True, False]

grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}



rf = RandomForestRegressor()

rf_search = RandomizedSearchCV(estimator = rf, param_distributions = grid, n_iter = 180, cv = 3, verbose=2, random_state=42, n_jobs = -1)



rf_search.fit(X_train, y_train)