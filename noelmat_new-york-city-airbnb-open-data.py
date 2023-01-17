import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing

import os

import seaborn as sns

import matplotlib.image as mpimg

%matplotlib inline



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.head()
print('Total number of observations in the dataset : {}'.format(data.shape[0]))
data.dtypes
data.isnull().sum()
data = data.drop(['id','host_name','last_review'],axis=1)

data.head()
data.fillna({'reviews_per_month':0},inplace=True)

data.reviews_per_month.isnull().sum()
data.neighbourhood_group.unique()
data.neighbourhood.unique()
data.room_type.unique()
#lets see which host ids have the most listings on Airbnb

top_host = data.host_id.value_counts().head(10)

top_host
#coming back to our dataset, we can confirm out findings with already existing column called 'calculated_host_listings_count'

top_host_check = data.calculated_host_listings_count.max()

top_host_check
#setting the figsize for future visualizations

sns.set(rc={'figure.figsize':(10,8)})
viz_1 = top_host.plot(kind='bar')

viz_1.set_title('Hosts with the most listings in NYC')

viz_1.set_ylabel('Count of listings')

viz_1.set_xlabel('Host IDs')

viz_1.set_xticklabels(viz_1.get_xticklabels(), rotation=45)
data.groupby('neighbourhood_group')['price'].describe().T
sub_df = data[data.price <500]



viz_2 = sns.violinplot(data=sub_df,x='neighbourhood_group',y='price')

viz_2.set_title('Density and distribution of prices for each neighbourhood_group')
# as we saw earlier there are too many unique values in neighbourhood columns. So lets get the top 10.



top_nbh = data.neighbourhood.value_counts().head(10).index
#now lets combine the neighbourhoods and room types for richer visualizations

subdf_1 = data.loc[data['neighbourhood'].isin(top_nbh)]



viz = sns.catplot(x='neighbourhood',hue='neighbourhood_group',col='room_type',data=subdf_1,kind='count')

viz.set_xticklabels(rotation=90)

plt.show()
#lets see what we can do with our given longitude and latitude columns



#lets check how the scatterplot will come out

viz_3 = sub_df.plot(kind='scatter',x='longitude',y='latitude',label='availability_365', c='price',cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(10,8))

viz_3.legend()
import urllib



plt.figure(figsize=(10,8))



i= urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/e/ec/Neighbourhoods_New_York_City_Map.PNG')

nyc_img=plt.imread(i)



plt.imshow(nyc_img,zorder=0,extent=[-74.258,-73.7,40.49,40.92])

ax=plt.gca()



sub_df.plot(kind='scatter',x = 'longitude',y='latitude',label='availability_365',c='price',ax=ax,cmap=plt.get_cmap('jet'),colorbar=True,alpha=0.4,zorder=5)

plt.legend()

plt.show()
#lets check the names columns now



names =[]



for name in data.name:

    names.append(name)

    

def split_names(name):

    spl=str(name).split()

    return spl



names_for_count=[]



for x in names:

    for word in split_names(x):

        word = word.lower()

        names_for_count.append(word)
from wordcloud import WordCloud



wc = WordCloud(width=800,height=800,background_color='white',min_font_size=10).generate(str(names_for_count))

plt.imshow(wc)

plt.axis('off')

plt.tight_layout(pad=0)

plt.show()
#lastly we will look at 'number_of_reviews'



#lets grab 10 most reviewed listings in NYC

top_reviewed_listings=data.nlargest(10,'number_of_reviews')

top_reviewed_listings
price_avg = top_reviewed_listings.price.mean()

print('Average price per night: {}'.format(price_avg))