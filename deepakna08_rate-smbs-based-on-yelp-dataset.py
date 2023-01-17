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
business.head(5)
business.city.value_counts().head(10)
business[['name', 'review_count', 'city', 'stars']].sort_values(ascending=False, by="review_count")[0:10]
business_cit = business[business.city.str.contains('Phoenix|Las Vegas|Toronto|Charlotte|Scottsdale',na=False)]
business_cat = business_cit[business_cit.categories.str.contains('Restaurants|Food|Hotel',na=False)]
city_business_reviews = business_cat.groupby(['city'])\
.agg({'business_id':'count', 'review_count': 'sum', 'stars': 'mean'}).sort_values(by='review_count', ascending=False)

city_business_reviews.head(5)       
business_cit = business[business.city.str.contains('Phoenix|Las Vegas|Toronto|Charlotte|Scottsdale',na=False)]
business_cat = business_cit[business_cit.categories.str.contains('Hair|Salons|Beauty|Grooming|Parlour|Stylists|Spa',na=False)]
city_business_reviews = business_cat.groupby(['city'])\
.agg({'business_id':'count', 'review_count': 'sum', 'stars': 'mean'}).sort_values(by='review_count', ascending=False)

city_business_reviews.head(5)    
business_cit = business[business.city.str.contains('Phoenix|Las Vegas|Toronto|Charlotte|Scottsdale',na=False)]
business_cat = business_cit[business_cit.categories.str.contains('Hair|Salons|Beauty|Grooming|Parlour|Stylists|Spa',na=False)]
#city_business_reviews = business_cat[business_cat['name'].count() > 3].groupby(['city','name'])

city_business_reviews = business_cat.groupby('name')['city'].size().reset_index(name='Count')

city_business_reviews[(city_business_reviews.Count>4) & (city_business_reviews.Count<16)].head(20).sort_values(by='Count', ascending=False)       
business_cit = business[business.city.str.contains('Phoenix|Las Vegas|Toronto|Charlotte|Scottsdale',na=False)]
business_cat = business_cit[business_cit.categories.str.contains('Hair|Salons|Beauty|Grooming|Parlour|Stylists|Spa',na=False)]
#city_business_reviews = business_cat[business_cat['name'].count() > 3].groupby(['city','name'])

city_business_reviews = business_cat.groupby('name')['city'].size().reset_index(name='Count')

city_business_reviews[(city_business_reviews.Count>15) & (city_business_reviews.Count<51)].head(20).sort_values(by='Count', ascending=False)      
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
reviews = pd.read_csv('../input/yelp_review.csv')
reviews.shape, tip.shape #there are 5.26 million reviews! 1 million tips
reviews.head(5)
tip.head()
