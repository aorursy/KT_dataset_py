# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
#import warnings
#warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_columns', 100)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
business = pd.read_csv('../input/yelp_business.csv')
business.head()
business.describe()
business_hours = pd.read_csv("../input/yelp_business_hours.csv")
business_hours.head()
business.columns
business.shape
#Null Values...
business.isnull().sum().sort_values(ascending=False)
#are all business Id's unique?
business.business_id.is_unique #business_id is all unique
business.city.value_counts()
business[['name', 'review_count', 'city', 'stars']].sort_values(ascending=False, by="review_count")[0:50]
city_business_counts = business[['city', 'business_id']].groupby(['city'])\
['business_id'].agg('count').sort_values(ascending=False)
city_business_counts = pd.DataFrame(data=city_business_counts)
city_business_counts.rename(columns={'business_id' : 'number_of_businesses'}, inplace=True)
city_business_counts[0:50].sort_values(ascending=False, by="number_of_businesses")\
.plot(kind='barh', stacked=False, figsize=[10,10], colormap='winter')
plt.title('Top 50 cities by businesses listed')
city_business_reviews = business[['city', 'review_count', 'stars']].groupby(['city']).\
agg({'review_count': 'sum', 'stars': 'mean'}).sort_values(by='review_count', ascending=False)
city_business_reviews.head(10)
city_business_reviews['review_count'][0:50].plot(kind='barh', stacked=False, figsize=[10,10], \
                                                 colormap='winter')
plt.title('Top 50 cities by reviews')
city_business_reviews[city_business_reviews.review_count > 50000]['stars'].sort_values()\
.plot(kind='barh', stacked=False, figsize=[10,10], colormap='winter')
plt.title('Cities with greater than 50k reviews ranked by average stars')
business['stars'].value_counts()
sns.distplot(business.stars, kde=False)
business['is_open'].value_counts()
tip = pd.read_csv('../input/yelp_tip.csv')
tip.head(10)
tip.shape
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 
                  'awful', 'wow', 'hate']
selected_words

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(vocabulary=selected_words, lowercase=False)
#corpus = ['This is the first document.','This is the second second document.']
#print corpus
selected_word_count = vectorizer.fit_transform(tip['text'].values.astype('U'))
vectorizer.get_feature_names()
word_count_array = selected_word_count.toarray()
word_count_array.shape
word_count_array.sum(axis=0)
temp = pd.DataFrame(index=vectorizer.get_feature_names(), \
                    data=word_count_array.sum(axis=0)).rename(columns={0: 'Count'})
temp.plot(kind='bar', stacked=False, figsize=[7,7], colormap='winter')
business[(business['city'] == 'Las Vegas') & (business['stars'] == 4.5)]
business[business.name=='"Earl of Sandwich"']
# This is where  have been to :)
business.loc[139699,:]
earl_of_sandwich = tip[tip.business_id==business.loc[139699,:].business_id]
earl_of_sandwich_selected_word_count = \
vectorizer.fit_transform(earl_of_sandwich['text'].values.astype('U'))
word_count_array = earl_of_sandwich_selected_word_count.toarray()
temp = pd.DataFrame(index=vectorizer.get_feature_names(), \
                    data=word_count_array.sum(axis=0)).rename(columns={0: 'Count'})
temp
temp.plot(kind='bar', stacked=False, figsize=[7,7], colormap='winter')
business[['name', 'review_count', 'city', 'stars']]\
[business.review_count>1000].sort_values(ascending=True, by="stars")[0:15]
business[business['name'] == '"Luxor Hotel and Casino Las Vegas"']
luxor_hotel = tip[tip.business_id==business.loc[6670,:].business_id]
luxor_hotel.info()
luxor_hotel_selected_word_count = vectorizer.fit_transform(luxor_hotel['text'].values.astype('U'))
word_count_array = luxor_hotel_selected_word_count.toarray()
temp = pd.DataFrame(index=vectorizer.get_feature_names(), \
                    data=word_count_array.sum(axis=0)).rename(columns={0: 'Count'})
temp.plot(kind='bar', stacked=False, figsize=[10,5], colormap='winter')
reviews = pd.read_csv('../input/yelp_review.csv')
reviews.shape, tip.shape #there are 5.26 million reviews! 1 million tips
reviews.head(5)
tip.head()
selected_words = ['sushi', 'miso', 'teriyaki', 'tempura', 'udon', \
                  'soba', 'ramen', 'yakitori', 'izakaya']


