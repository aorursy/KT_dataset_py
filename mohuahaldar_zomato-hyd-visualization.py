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
tr_data=pd.read_csv('/kaggle/input/zomato-restaurants-hyderabad/Restaurant reviews.csv')

tr_data
tr_data.Review.isnull().sum()
tr_data.Rating.unique()
tr_data[tr_data.Rating=='Like']

tr_data.drop(tr_data.index[7601], inplace=True)
rows=tr_data[tr_data.Rating.isnull()].Restaurant.index

tr_data.drop(rows, inplace=True)
tr_meta=pd.read_csv('/kaggle/input/zomato-restaurants-hyderabad/Restaurant names and Metadata.csv')

tr_meta.head()
tr_meta.isnull().sum()
tr_meta.shape
tr_meta.info()
tr_meta.Cost.unique()
tr_meta.Cost=tr_meta.Cost.str.replace(',','').astype('int64')

tr_meta.Cost
cuisines=[]

[cuisines.extend(s.split(','))for s in tr_meta.Cuisines.unique()]

from collections import Counter

all_cuisine=[]

all_cuisine.extend(s.strip() for s in cuisines)

all_cuisine=Counter(all_cuisine)

cuisine_df=pd.DataFrame.from_dict(all_cuisine, orient='index')

cuisine_df.columns=['Count']



import matplotlib.pyplot as plt

import seaborn as sns
plt.figure(figsize=(18,12))

g=sns.barplot(cuisine_df.index, cuisine_df.Count)

g.set_xticklabels(cuisine_df.index,rotation=30)

plt.xlabel('Cuisine')

plt.ylabel('How many restaurants serving')

plt.show()
counts=[len(s) for s in tr_meta.Cuisines.str.split(',')]

df_costofCuisine=pd.DataFrame(counts)

df_costofCuisine.columns=['No_of_cuisines_offered']

df_costofCuisine['Cost']=tr_meta.Cost

sns.boxplot(df_costofCuisine['No_of_cuisines_offered'], df_costofCuisine['Cost'])

plt.show()

df_merged=tr_data.merge(tr_meta, how='inner', left_on='Restaurant', right_on='Name')

df_merged.head()
df_merged.Rating.unique()


sns.boxplot(df_merged.Rating, df_merged.Cost)

plt.show()
tr_data[['Restaurant','Rating']].groupby('Restaurant').count()
indices_null=tr_data[tr_data.Review.isnull()].index

print(len(tr_data.Restaurant.unique()))

tr_data.drop(indices_null, inplace=True)



!pip install nltk==3.4
import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

def cleanText(review):

    StopWords = set(stopwords.words('english'))

    #review_data=review.lower()

    tokens=[token for token in word_tokenize(review) if token not in StopWords and len(token)>3]

    return tokens
tr_data['Review']=tr_data.Review.apply(lambda row: row.lower())



tr_data['CleanData']=tr_data['Review'].apply(lambda row:cleanText(row))
def findBigrams(cleanReview):

    bigm=[]

            

    if cleanReview:        

        bigm=list(nltk.bigrams(cleanReview))    

     

       

    return bigm
tr_data['Bigrams']=tr_data['CleanData'].apply(lambda row:findBigrams(row))

rating_bigrams={}

for grp, data in tr_data.groupby('Rating'):

    bigrams=[]

    for d in data['Bigrams']:

        bigrams.extend(d)

    common=nltk.FreqDist(bigrams).most_common(7)

    rating_bigrams[grp]=common    

   

    bigrams.clear()

bi_rat