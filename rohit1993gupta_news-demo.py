# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as py

import seaborn as sns

from nltk.corpus import stopwords as stop

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import pairwise_distances

from sklearn.feature_extraction.text import TfidfVectorizer
df = pd.read_json("/kaggle/input/news-category-dataset/News_Category_Dataset_v2.json", lines=True)
df.head()
df.describe()
df.info()
df = df[df['date']>pd.Timestamp(2018,1,1)]
df.shape
df = df[df['headline'].apply(lambda x:len(x.split())>5)]

df.shape
df.sort_values('headline', inplace=True, ascending=False)

print(df.shape)

df_duplicated=df.duplicated('headline',keep=False)

print(df.shape)

df = df[~df_duplicated]

print(df.shape)
# lets check if any of the cell is empty or has an ambigiious value

df.isna().sum()
df.describe()
index = df['category'].value_counts().index

index.shape
values = df['category'].value_counts().values

type(values)
py.bar(df['category'].value_counts().index, df['category'].value_counts(),width=0.8)

py.show()
news_per_month= df.resample('M', on= "date")['headline'].count()

news_per_month
py.figure()

py.title('Month wise distribution')

py.xlabel('Month')

py.ylabel('Number of articles')

py.bar(news_per_month.index.strftime('%b'), news_per_month, width=0.8)
sns.distplot(df['headline'].str.len(), hist=False)
df['day and month']= df['date'].dt.strftime("%a")+'_'+df['date'].dt.strftime('%b')
df.index= range(df.shape[0])

df_temp=df.copy()
stop_words = stop.words('english')
for i  in range(len(df_temp["headline"])):

    string=""

    for word in df_temp["headline"][i].split():

        word = ("".join(e for e in word if e.isalpha()))

        word = word.lower()

        if not word in stop_words:

            string += word +" " 

    if i%500 == 0:

        print(i)

    df_temp.at[i,'headline']= string.strip()
from nltk.stem import WordNetLemmatizer

lemitizer = WordNetLemmatizer()
for i in range(len(df_temp['headline'])):

    string=""

    for w in df_temp['headline'][i]:

        string += lemitizer.lemmatize(w,pos='v')+" "

    print(string)

    df_temp.at[i,'headline'] += string.strip()
df_temp['headline'][0]
vectorize = CountVectorizer()

vectorize_features = vectorize.fit_transform(df_temp['headline'])

vectorize_features.shape
pd.set_option('display.max_colwidth', -1)



# to get the biggest possible headline to display
def bag_of_words_model(row_index, output_values):

    # to find the distance of the featutres of row_index corresponding to all the other rows

    couple_dist =  pairwise_distances(vectorize_features, vectorize_features[row_index])

    indices = np.argsort(couple_dist.ravel())[0:output_values]

    df1 = pd.DataFrame(

        {'publish_date':df['date'][indices].values, 

         'headline':df['headline'][indices].values,

         'euclidean distance':couple_dist[indices].ravel()

        }

    )

    print("headline: ",df['headline'][indices[0]])

    return (df1.iloc[1:,])



bag_of_words_model(133, 11)
vectorizer = TfidfVectorizer(min_df=0)

tfidf_headline_features = vectorize.fit_transform(df_temp['headline'])
def tfidf_based_model(row_index, num_similar_items):

    couple_dist = pairwise_distances(tfidf_headline_features,tfidf_headline_features[row_index])

    indices = np.argsort(couple_dist.ravel())[0:num_similar_items]

    df1 = pd.DataFrame({'publish_date': df['date'][indices].values,

               'headline':df['headline'][indices].values,

                'Euclidean similarity with the queried article': couple_dist[indices].ravel()})

    print("="*30,"Queried article details","="*30)

    print('headline : ',df['headline'][indices[0]])

    print("\n","="*25,"Recommended articles : ","="*23)

    

    #return df.iloc[1:,1]

    return df1.iloc[1:,]



tfidf_based_model(133, 11)