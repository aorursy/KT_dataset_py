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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from wordcloud import WordCloud,STOPWORDS #for visual representation of text data.

#Stopwords are common words which do not provide any reasonable value to our data, e.g it, the, are. we
df=pd.read_csv('../input/515k-hotel-reviews-data-in-europe/Hotel_Reviews.csv') #loading the datasets
df.head() # Checking the top 5 rows

df.sample(5) # Just sampling out any 5 rows for a better look
df.shape # 515,738 rows and 17 columns

df.info() # Shows the datatype of the columns
df.describe()
df.isnull().sum() # Checking missing values, Latitude and longitude has some missing values (3268)
df['Hotel_Address'].nunique()



# Countries in this dataset involves France, United Kingdom, Netherlands, Spain, Italy, Netherlands, Austria

# 1493 unique hotel address
df.columns # Prints 17 unique columns
# Plotting the Average scores of the hotels

df_sd = df[['Hotel_Name','Average_Score']].drop_duplicates() # Dropping any duplicates

plt.figure(figsize = (14,6))

sns.countplot(x = 'Average_Score',data = df_sd,color = 'green')

# From the graph below, we can notice that most hotels were given scores ranging from 8.1 to 8.9
df.Average_Score.describe()

# There are 34 unique average scores

# Minimum Average score is 5.2

# Maximum Average score is 9.8

# 25% of the hotels have an Average_score of 8.1 - 5.2

# 50% of the hotels have an Average_score of 8.4 - 8.2

# 75% of the hotels have an Average_score of 8.8 - 8.5
# We check out the distribution of hotels in the European countries

df.Hotel_Address = df.Hotel_Address.str.replace('United Kingdom','UK') # Replacing 'united kingdom' with 'UK' for easy use

df['EC'] = df.Hotel_Address.apply(lambda x: x.split(' ')[-1]) # Splitting the hotel address and picking out the last string which would be the countries

#Plotting with matplotlib 

plt.figure(figsize = (12,5))

plt.title('Hotel distribution in European countries')

df.EC.value_counts().plot.barh(color = 'green')
df[df.Average_Score >= 8.8][['Hotel_Name','Average_Score','Total_Number_of_Reviews']].drop_duplicates().sort_values(by ='Total_Number_of_Reviews',ascending = False)[:10]

# We now attempt to find the 10 most popular hotels based on 'Total number of reviews, Average score greater than 8.8, and the Hotel names'
df['Positive_Review'] # Having a look at positive reviews
import nltk # Natural language processing toolkit

from nltk import FreqDist # Frequency distribution



import re # for regular expressions

import spacy # library for advanced Natural Language Processing 



# function to plot most frequent terms

def freq_words(x, terms = 30):

  all_words = ' '.join([text for text in x])

  all_words = all_words.split()



  fdist = FreqDist(all_words)

  words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})



  # selecting top 20 most frequent words

  d = words_df.nlargest(columns="count", n = terms) 

  plt.figure(figsize=(20,5))

  ax = sns.barplot(data=d, x= "word", y = "count")

  ax.set(ylabel = 'Count')

  plt.show()
freq_words(df['Positive_Review']) # Frequency distribution of common words in positive reviews
freq_words(df['Negative_Review'])  # Frequency distribution of common words in negative reviews
# You probably noticed we has a lot of word like 'the', 'was', 'to' e.t.c which won't help so we would remove them.

# First of all, we remove unwanted characters, numbers and symbols

df['Positive_Review'] = df['Positive_Review'].str.replace("[^a-zA-Z#]", " ")

df['Negative_Review'] = df['Negative_Review'].str.replace("[^a-zA-Z#]", " ")
from nltk.corpus import stopwords

stop_words = stopwords.words('english')

# function to remove stopwords

def remove_stopwords(rev):

    rev_new = " ".join([i for i in rev if i not in stop_words])

    return rev_new

# I would apply everyting below to both positive and negative reviews

# remove short words (length < 4)

df['Positive_Review'] = df['Positive_Review'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

df['Negative_Review'] = df['Negative_Review'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))





# remove stopwords from the text

reviews_1 = [remove_stopwords(r.split()) for r in df['Positive_Review']]

reviews_2 = [remove_stopwords(r.split()) for r in df['Negative_Review']]







# make entire text lowercase

#reviews_1= [r.lower() for r in reviews_1] 

#reviews_2= [r.lower() for r in reviews_2]

# From what i read, Nltk sees 'stop' and 'STOP' as different things. Making all lowercase seems good but I dont think i want to so that i can identify the 'No Negative' and 'No positive'in it. 

# if you dont get, just bring the question to our channel
freq_words(reviews_1, 30) # Checking frequency of most used words in positive reviews
freq_words(reviews_2, 30) # Checking frequency of most used words in negative reviews
# Using wordcloud to visually represent the text data

def wordcloud_draw(data, color = 'black'):

    words = ' '.join(data)

    cleaned_word = " ".join([word for word in words.split()

                            if 'http' not in word

                                and not word.startswith('@')

                                and not word.startswith('#')

                                and word != 'RT'

                            ])

    wordcloud = WordCloud(stopwords=STOPWORDS,

                      background_color=color,

                      width=2500,

                      height=2000

                     ).generate(cleaned_word)

    plt.figure(1,figsize=(13, 13))

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()

    

print("Positive reviews")

wordcloud_draw(reviews_1,'white')

print("Negative reviews")

wordcloud_draw(reviews_2)
