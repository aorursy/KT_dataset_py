import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('darkgrid')

import plotly.express as ex

import plotly.graph_objs as go

import plotly.figure_factory as ff

import bs4 as bs

import re
movies_data = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_movies.csv')

movies_data.drop(columns=['homepage','status'],inplace=True)

credits_data = pd.read_csv('/kaggle/input/tmdb-movie-metadata/tmdb_5000_credits.csv')

movies_data.head(5)
genres_list =[]

keywords_list=[]

production_companies_list=[]

production_countries_list = []

spoken_languages_list = []

for index,row in  movies_data.iterrows():

    dec = re.findall(r'"([A-Za-z0-9_\./\\-]*)"',row['genres'])

    dec = [cat for cat in dec if cat not in ['id','name']]

    genres_list +=dec

    dec = re.findall(r'"([A-Za-z0-9_\./\\-]*)"',row['keywords'])

    dec = [cat for cat in dec if cat not in ['id','name']]

    keywords_list +=dec

    dec = re.findall(r'"([A-Za-z0-9_\./\\-]*)"',row['production_companies'])

    dec = [cat for cat in dec if cat not in ['id','name']]

    production_companies_list +=dec

    dec = re.findall(r'"([A-Za-z0-9_\./\\-]*)"',row['production_countries'])

    dec = [cat for cat in dec if cat not in ['id','name']]

    production_countries_list +=dec

    dec = re.findall(r'"([A-Za-z0-9_\./\\-]*)"',row['spoken_languages'])

    dec = [cat for cat in dec if cat not in ['id','name'] and len(cat)<= 2]

    spoken_languages_list +=dec



genres_list = list(set(genres_list))

keywords_list = list(set(keywords_list))

production_companies_list = list(set(production_companies_list))

production_countries_list = list(set(production_countries_list))

spoken_languages_list = list(set(spoken_languages_list))





movies_data.release_date = pd.to_datetime(movies_data.release_date)



movies_data['Day_Of_Week'] = movies_data.release_date.apply(lambda x: x.weekday())

movies_data['Month'] = movies_data.release_date.apply(lambda x: x.month)

movies_data['Year'] = movies_data.release_date.apply(lambda x: x.year)

movies_data.drop(columns=['release_date'],inplace=True)
movies_data = movies_data.dropna()
info = movies_data.describe()

info.loc['skew'] = movies_data.skew()

info.loc['kurt'] = movies_data.kurt()

info
average_year = movies_data.groupby(by='Year').mean().reset_index()

average_year = average_year.drop(columns=['id','Day_Of_Week','Month'])



fig,axs = plt.subplots(2,3,'all')

fig.set_figwidth(25)

fig.set_figheight(15)

c,r=0,0

for col in average_year.columns[1:]:

    sns.lineplot(data=average_year,x='Year',y=col,ax=axs[r,c],label='Mean '+col)

    axs[r,c].set_title('Average '+col+' over the years',fontsize=15)

    if c==2:

        r+=1

        c=0

    else:

        c+=1
from matplotlib.ticker import FormatStrFormatter

plt.figure(figsize=(20,11))

ax = sns.kdeplot(movies_data['budget'],label='Budget')

ax = sns.kdeplot(movies_data['revenue'],label='Revenue')

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax.set_title('Movie Budget Over Revenue Distribution',fontsize=19)
plt.figure(figsize=(20,11))

ax = sns.distplot(movies_data['popularity'])

ax.set_title('Movie Popularity Distribution',fontsize=19)
movies_data = movies_data.drop(movies_data.query('Year == 2017').index)

movies_data = movies_data.drop(movies_data.query('revenue == 0').index)

movies_data = movies_data.drop(movies_data.query('budget == 0').index)

movies_data.revenue = np.log(movies_data.revenue)

movies_data.budget = np.log(movies_data.budget)

movies_data = movies_data.query('revenue > 10')

movies_data = movies_data.query('budget > 10')

from matplotlib.ticker import FormatStrFormatter

plt.figure(figsize=(20,11))

ax = sns.kdeplot(movies_data['budget'],label='Budget')

ax = sns.kdeplot(movies_data['revenue'],label='Revenue')

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

ax.set_title('Movie Budget Over Revenue Distribution After Normallization And Outlier Removal',fontsize=19)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

vectorizer = TfidfVectorizer()



tagline = vectorizer.fit_transform(movies_data.tagline)

tagline



tsvd = TruncatedSVD(n_components=900)

tagline = tsvd.fit_transform(tagline)
cum_sum = np.cumsum(tsvd.explained_variance_ratio_)

plt.figure(figsize=(20,11))

ax= sns.lineplot(x=np.arange(0,len(cum_sum)),y=cum_sum)

ax.set_title('Cumulative Variance Ratio',fontsize=20)

ax.set_xlabel('Number Of Components',fontsize=16)

ax.set_ylabel('Explained Variance',fontsize=16)

plt.show()
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression



def RMSE(Y,Y_HAT):

    return np.sqrt(mean_squared_error(Y,Y_HAT))
tag_df =pd.DataFrame(tagline)



train_x,test_x,train_y,test_y = train_test_split(tag_df,movies_data.revenue)

rf_pipe =Pipeline(steps=[('model',LinearRegression())])

rf_pipe.fit(train_x,train_y)

predictions= rf_pipe.predict(test_x)
print('Test Set RMSE: ',RMSE(predictions,test_y))
rf_pipe =Pipeline(steps=[('model',LinearRegression())])

rf_pipe.fit(tag_df,movies_data.revenue)

predictions= rf_pipe.predict(tag_df)

print('Entire Data RMSE: ',RMSE(predictions,movies_data.revenue))
plt.figure(figsize=(20,11))

ax = sns.lineplot(x=np.arange(0,len(movies_data.revenue)),y=movies_data.revenue,label='Actual Revenue')

ax = sns.lineplot(x=np.arange(0,len(movies_data.revenue)),y=predictions,label='Predicted Revenue')

ax.set_xlabel('Sample Number',fontsize=16)

ax.set_ylabel('Log(Revenue)',fontsize=16)

ax.set_title("Prediction VS Real Values",fontsize=20)

plt.show()
plt.figure(figsize=(20,11))

ax = sns.distplot(movies_data.revenue,label='Actual Revenue')

ax = sns.distplot(predictions,label='Predicted Revenue')

ax.set_xlabel('Log(Revenue)',fontsize=16)

ax.set_ylabel('Density',fontsize=16)

ax.set_title("Prediction VS Real Values",fontsize=20)

plt.plot([np.mean(predictions),np.mean(predictions)],[0,0.7],c='r',label='Prediction Mean')

plt.plot([np.mean(predictions),np.mean(movies_data.revenue)],[0,0.5],c='g',label='Actual Values Mean')

plt.legend(prop={'size':18})





plt.show()