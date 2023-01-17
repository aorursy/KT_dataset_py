import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
# Load the data

debate_df=pd.read_csv("../input/us-presidential-debatefinal-october-2020/Presidential_ debate_USA_ final_October_ 2020.csv")
debate_df.info


debate_df=debate_df.drop(debate_df.columns[0],axis=1)

debate_df['length']=debate_df['speech'].apply(len)

#debate_df



debate_df.describe()
TRUMP_debate_df = debate_df[(debate_df['speaker']=='TRUMP')]

BIDEN_debate_df = debate_df[(debate_df['speaker']=='BIDEN')]
TRUMP_debate_df.describe()
BIDEN_debate_df.describe()
plt.rcParams['figure.figsize'] = (15, 5)



plt.subplot(1, 2, 1)

plt.hist(TRUMP_debate_df['length'], bins=30, color='#ff0422',alpha=0.7)

plt.title('TRUMP', fontsize = 20)

plt.xlabel('Letters', fontsize = 15)

plt.ylabel('count', fontsize = 15)







plt.subplot(1, 2, 2)

plt.hist(BIDEN_debate_df['length'], bins=30, color='#0504aa',alpha=0.7)

plt.title('BIDEN', fontsize = 20)

plt.xlabel('Letters', fontsize = 15)

plt.ylabel('count', fontsize = 15)



plt.show()


debate_df['TOPIC']=debate_df['TOPIC'].str.replace('TOPIC: ','')

debate_df_by_topic_TRUMP = debate_df[debate_df['speaker']=='TRUMP'].groupby(["TOPIC"])["length"].sum().reset_index()

debate_df_by_topic_BIDEN = debate_df[debate_df['speaker']=='BIDEN'].groupby(["TOPIC"])["length"].sum().reset_index()

# plotting a pie chart for browsers



plt.rcParams['figure.figsize'] = (18, 7)

plt.subplot(1, 2, 1)

plt.pie(debate_df_by_topic_TRUMP.length,labels=debate_df_by_topic_TRUMP.TOPIC, autopct = '%.2f%%',startangle = 90)

plt.title('TRUMP', fontsize = 20)

plt.axis('off')

plt.legend()



# plotting a pie chart for browsers

plt.subplot(1, 2, 2)

plt.pie(debate_df_by_topic_BIDEN.length,labels=debate_df_by_topic_BIDEN.TOPIC, autopct = '%.2f%%',startangle = 90)

plt.title('BIDEN', fontsize = 20)

plt.axis('off')

plt.legend()

plt.show()
TRUMP_Speech = debate_df[(debate_df['speaker']=='TRUMP')]

BIDEN_Speech = debate_df[(debate_df['speaker']=='BIDEN')]





TRUMP_sentences=TRUMP_Speech['speech'].tolist()

BIDEN_sentences=BIDEN_Speech['speech'].tolist()



TRUMP_sentences="".join(TRUMP_sentences)

BIDEN_sentences="".join(BIDEN_sentences)



import nltk

from nltk.corpus import stopwords

nltk.download('stopwords')

from string import punctuation

from nltk.tokenize import word_tokenize



all_stopwords = stopwords.words('english')

#########################################

sw_list = ['know','\'re','n\'t','going','-','_','—','’','‘','us','say','said']

all_stopwords.extend(sw_list)

#########################################



def remove_stopwords(text):

    text_tokens = word_tokenize(text)

    tokens_without_sw = [word for word in text_tokens if not word in all_stopwords]



    return " ".join(tokens_without_sw)



def remove_punctuation(text):

    return ''.join(c for c in text if c not in punctuation)

TRUMP_sentences=remove_stopwords(TRUMP_sentences)

BIDEN_sentences=remove_stopwords(BIDEN_sentences)



TRUMP_sentences=remove_punctuation(TRUMP_sentences)

BIDEN_sentences=remove_punctuation(BIDEN_sentences)
from wordcloud import WordCloud 



plt.rcParams['figure.figsize'] = (20, 5)



plt.subplot(1, 2, 1)

wordcloud = WordCloud().generate(TRUMP_sentences)

plt.imshow(wordcloud)

plt.axis("off")

plt.title('TRUMP', fontsize = 20)



plt.subplot(1, 2, 2)

wordcloud = WordCloud().generate(BIDEN_sentences)

plt.imshow(wordcloud)

plt.axis("off")

plt.title('BIDEN', fontsize = 20)

plt.show()