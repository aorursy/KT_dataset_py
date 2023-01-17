import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
reviews_df= pd.read_csv("/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv", delimiter = '\t')

reviews_df.head()
print(reviews_df.info())
reviews_df.describe()
reviews_df.hist(bins = 30, figsize = (13,5), color = 'g')
plt.show()
reviews_df['length']= reviews_df['verified_reviews'].apply(len)
reviews_df.head()
reviews_df['length'].plot(kind= 'hist', bins= 40)

reviews_df['length'].describe()
reviews_df['verified_reviews'][reviews_df.length==2851]
reviews_df[reviews_df['length'] == 2851]['verified_reviews'].iloc[0]
reviews_df[reviews_df['length'] == 150]['verified_reviews'].iloc[0]
feedback_breakup=reviews_df.groupby(reviews_df.feedback).count()
sns.countplot(reviews_df.feedback)
sns.countplot(x= 'rating', data= reviews_df )
plt.figure(figsize= (8,6))
sns.catplot(y= 'variation', data= reviews_df, order= reviews_df['variation'].value_counts().index, col= 'feedback', kind= 'count')
plt.figure(figsize= (15,10))
sns.catplot(y= 'variation',data= reviews_df, order= reviews_df['variation'].value_counts().index, col= 'rating', kind= 'count')
negative = reviews_df[reviews_df['feedback']==0]
positive = reviews_df[reviews_df['feedback']==1]
    
negativesentence=negative.verified_reviews.tolist()
fullnegative=" ".join(negativesentence)
fullnegative
positivesentence= positive.verified_reviews.tolist()
fullpositive= " ".join(positivesentence)
fullpositive
from wordcloud import WordCloud

plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(fullnegative))
plt.figure(figsize=(20,20))
plt.imshow(WordCloud().generate(fullpositive))
