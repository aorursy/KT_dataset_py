import os
os.listdir("../input")
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import random 

data=pd.read_csv("../input/tweet.csv")
data.dropna()
df = data['text']



stopword = ['english','is','and','it','the','on','of','pri','for','not','to','in','are','be','zenoss'] 

count_vect = CountVectorizer(max_df=0.8, min_df=2, stop_words=stopword)
doc_term_matrix = count_vect.fit_transform(df.values.astype('U')) 
LDA = LatentDirichletAllocation(n_components=10, random_state=42)
LDA.fit(doc_term_matrix)

first_topic = LDA.components_[0]
for i,topic in enumerate(LDA.components_):
    print(f'Top 10 words for topic #{i}:')
    print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')

topic_values = LDA.transform(doc_term_matrix) 
df['Topic'] = topic_values.argmax(axis=1)