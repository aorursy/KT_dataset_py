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
import matplotlib.pyplot as plt
import matplotlib as mpl
import nltk.classify.util
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.classify import NaiveBayesClassifier
import numpy as np
import re
import string
import nltk
%matplotlib inline

df = pd.read_csv("../input/consumer-reviews-of-amazon-products/1429_1.csv")
low_memory=False
df.head()
df.columns
from IPython.display import HTML
cat_hist = df.groupby('categories',as_index=False).count()
HTML(pd.DataFrame(cat_hist['categories']).to_html())
df_new = df[[ 'name', 'brand',
       'categories', 'reviews.doRecommend',
       'reviews.numHelpful', 'reviews.rating', 
       'reviews.text', 'reviews.title', 'reviews.username']]
df_new.head(5)
import csv
from collections import Counter
filename='/kaggle/input/consumer-reviews-of-amazon-products/1429_1.csv'
with open(filename, 'r') as f:
    column = (row[17] for row in csv.reader(f))
    print("Most frequent value: {0}".format(Counter(column).most_common()[0][0]))
from collections import Counter
filename='/kaggle/input/consumer-reviews-of-amazon-products/1429_1.csv'
with open(filename, 'r') as f:
    column = (row[14] for row in csv.reader(f))
    print("Most frequent value: {0}".format(Counter(column).most_common()[0][0]))
df_reviews = df[[ 'reviews.rating','reviews.text', 'reviews.title',]]
df_reviews.head(5)
df_classify = df_reviews[df_reviews["reviews.rating"].notnull()]
df_classify["sentiment"] = df_classify["reviews.rating"] >= 4
df_classify["sentiment"] = df_classify["sentiment"].replace([True , False] , ["Postive" , "Negative"])

# Lets count positive and negative review
df_classify["sentiment"].value_counts().plot.bar()
reviews=pd.DataFrame(df.groupby('reviews.rating').size().sort_values(ascending=False).rename('No of Users').reset_index())
review.head()
import pandas as pd
add = "../input/consumer-reviews-of-amazon-products/1429_1.csv"
reviews = pd.read_csv(add,low_memory=False)
reviews.columns = ['id', 'name', 'asins', 'brand', 'categories', 'keys', 'manufacturer','date', 'dateAdded', 'dateSeen',
       'didPurchase', 'doRecommend', 'id','numHelpful', 'rating', 'sourceURLs','text', 'title', 'userCity',
       'userProvince', 'username']
reviews.nunique()
reviews.isnull().sum()
reviews.drop(labels=['didPurchase','id','userCity','userProvince'],axis=1,inplace=True)
reviews.isnull().sum()
from matplotlib import pyplot
%matplotlib inline

star = reviews.rating.value_counts()
print("*** Rating distribution ***")
print(star)
star.sort_index(inplace=True)
star.plot(kind='bar',title='Amazon customer ratings',figsize=(6,6),style='Solarize_Light2')
NPS_score = round (100*((star.loc[5])-sum(star.loc[1:3]))/sum(star.loc[:]),2)
print (" NPS score of Amazon is : "  + str(NPS_score))
kindle = reviews[reviews.name=='Amazon Kindle Paperwhite - eBook reader - 4 GB - 6 monochrome Paperwhite - touchscreen - Wi-Fi - black,,,']

kindle.isnull().sum()
kindle_s = kindle.rating.value_counts()
kindle_s.sort_index(inplace=True)

Kindle_NPS_score = round (100*(kindle_s[5]-sum(kindle_s[1:3]))/sum(kindle_s),2)
print (" NPS score of Kindle is : "  + str(Kindle_NPS_score))
#better NPS than overall amazon
kindle_s.plot(kind='bar',title='Amazon customer ratings for Kindle',figsize=(6,6),style='Solarize_Light2')
kindle.doRecommend.value_counts()
kindle.rating.hist(by=kindle.doRecommend,figsize=(12,6))
