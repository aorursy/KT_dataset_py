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
df = pd.read_csv('/kaggle/input/nlp-topic-modelling/Reviews.csv')
df.head()
#Taking only the id and text column for Topic Modelling



df = df[['Id','Text']]
df.shape
#Use TF-IDF Vectorization to create a vectorized document term matrix.



from sklearn.feature_extraction.text import TfidfVectorizer



tfidf = TfidfVectorizer(max_df=0.95,min_df=2,stop_words='english')



dtm = tfidf.fit_transform(df['Text'])
dtm.shape
#Using Scikit-Learn create an instance of NMF with 20 expected components. (Using random_state=42)



from sklearn.decomposition import NMF



nmf_model = NMF(n_components=20 , random_state=42)



nmf_model.fit(dtm)
#Printing out the top 15 most common words for each of the 20 topics.



for index,topic in enumerate(nmf_model.components_):

    print(f'Top 15 words of the TOPIC #{index}')

    print([tfidf.get_feature_names()[i] for i in topic.argsort()[-15:]])

    print('\n')
# Adding a new column to the original quora dataframe that labels each question into one of the 20 topic categories



topic_results = nmf_model.transform(dtm)
df['Topic'] = topic_results.argmax(axis=1)
df.head()