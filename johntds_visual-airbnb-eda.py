

import numpy as np

from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

from sklearn import linear_model ,neighbors,preprocessing,svm,tree

from sklearn.neighbors import NearestNeighbors

from sklearn.preprocessing import StandardScaler,LabelEncoder

import pandas as pd

from IPython.display import display

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split

import warnings

from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt

from sklearn import linear_model , neighbors,preprocessing,svm,tree

from sklearn.linear_model import LogisticRegression,LinearRegression

from sklearn.neighbors import NearestNeighbors,KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score,accuracy_score,make_scorer

import seaborn as sns

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier,VotingClassifier,ExtraTreesClassifier,BaggingClassifier

from sklearn.linear_model import Lasso, ElasticNet, LinearRegression

import sys

from xgboost import XGBRegressor

pd.set_option('display.max_rows', 100)

pd.set_option('display.max_columns', 100)

# plt.style.use('dark_background')

import vecstack

from vecstack import stacking

from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

from scipy.stats import probplot

from itertools import combinations

from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import r2_score,accuracy_score,make_scorer,log_loss,precision_score

import seaborn as sns

from sklearn.model_selection import GridSearchCV,KFold,cross_val_score

from sklearn.neural_network import MLPClassifier

from sklearn.svm import NuSVC,SVC

from scipy import std ,mean

from scipy.stats import norm

from scipy import stats







data=pd.read_csv ('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')





from pylab import rcParams





sns.set_palette('dark')

plt.rcParams['figure.figsize'] = [13, 6.5] 

sns.set_context("paper", rc={"font.size":15,"axes.titlesize":22,"axes.labelsize":20,"xtick.labelsize":13,"ytick.labelsize":13,"legend.fontsize":12,"figure.suptitlesize":25}) 
data.drop(['id','host_name','last_review'], axis=1, inplace=True)



popular=data.host_id.value_counts()[data.host_id.value_counts()>50]





print('POPULAR HOSTS MEAN PRICE :',data[data.host_id.isin(popular.index)].price.mean(),' \n GENERAL MEAN PRICE:',data.price.mean(),

      ' \n Popular hosts take more money. Reasons:  \n 1) MORE houses so they increase the price \n 2)If they can afford more houses they can afford better quality houses too'

    );
plt.rcParams['figure.figsize'] = [14, 7] 



a=sns.boxplot(data.host_id[data.host_id.isin(popular.index)],data.price);

plt.xticks(rotation=45);

a.set(title='POP HOSTS');
popularData=data[data.calculated_host_listings_count>50]



a=sns.countplot(popularData.neighbourhood_group,palette='winter')

a.set(title='POP HOSTS');
fig,axes=plt.subplots(1,2)





sns.distplot(data[(data['minimum_nights'] <= 40) & (data['minimum_nights'] > 0)]['minimum_nights'], bins=40,ax=axes[0])





axes[0].set(title='GENERAL')

axes[1].set(title='POPULAR HOSTS')





sns.distplot(popularData[(popularData['minimum_nights'] <= 40) & (popularData['minimum_nights'] > 0)]['minimum_nights'], bins=40,ax=axes[1]);



fig,axes=plt.subplots(1,2,figsize=(16,8))

sns.distplot(data.availability_365,ax=axes[0])





sns.distplot(popularData.availability_365,ax=axes[1])



axes[0].set(title='GENERAL')

axes[1].set(title='POPULAR HOSTS');
print(

    'Pop hosts availability mean and median:', popularData.availability_365.mean(),

    popularData.availability_365.median(),

    

    "\n General mean and median:",data.availability_365.mean(), 

    

    data.availability_365.median(),

    

    

    '\n  Price and time spending of popular hosts on each house have impact here');


sns.distplot(data[data.price<500].price);
a=sns.boxplot(data.price)

a.set(xlim=(0,1000));
unique_vals = data['neighbourhood_group'].unique()

targets = [data.loc[data['neighbourhood_group'] == val] for val in unique_vals]

i=0

fig,axes=plt.subplots(1,len(unique_vals),figsize=(16,8))

for t in targets:   

  

    a=sns.distplot(t.price,ax=axes[i],hist=False) 

    a.set(xlim=(0,600),ylim=(0,0.013),xlabel=str(t.neighbourhood_group.iloc[0]))

    i+=1

   
a=sns.boxplot(x='neighbourhood_group', y='price',data=data[data.price<1000])

sns.countplot(data.neighbourhood_group);


plt.figure(figsize=(14,6))

sns.scatterplot(data.longitude,data.latitude,hue=data.neighbourhood_group);

plt.figure(figsize=(15,7))

plt.scatter(data.longitude, data.latitude, s=5,c=data.availability_365, cmap='winter' , alpha=0.9)

bar = plt.colorbar()

bar.set_label('AVAILABLE DAYS OF YEAR');
sns.barplot(data.neighbourhood_group,data.availability_365);
price = pd.cut(data.price[data['price']<=1000], 33)

a=sns.barplot(price,data[data.price<=1000].availability_365)

a.set_xticklabels(a.get_xticklabels(),rotation=80);

popNeighb=data.neighbourhood.value_counts().head(17)

a=sns.countplot(y=data[data.neighbourhood.isin(popNeighb.index)].neighbourhood ,orient='h',order=popNeighb.index,hue=data.neighbourhood_group)
data[data.neighbourhood.isin(popNeighb.index)].price.mean(),data.price.mean();


sns.countplot(x = 'room_type',hue = "neighbourhood_group",data = data)

plt.title("ROOM TYPES IN DIFFERENT AREAS");


a=sns.catplot(y='neighbourhood',orient='v', hue='neighbourhood_group', col='room_type', 

              

              data=data[(data.neighbourhood.isin(popNeighb.index))&(data.room_type!='Shared room')],kind='count')



a.fig.set_size_inches(17,10);


plt.figure(figsize=(15,8))

sns.barplot(data.room_type,data.price);

sns.scatterplot(data.number_of_reviews,data.price);
