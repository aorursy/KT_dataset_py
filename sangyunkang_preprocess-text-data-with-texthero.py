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
from zipfile import ZipFile
zip = ZipFile('/kaggle/input/spooky-author-identification/train.zip')

zip.extractall()
!ls
df = pd.read_csv('train.csv')
df.head()
!pip install texthero
from texthero import preprocessing

df['clean_text'] = preprocessing.clean(df['text'])
preprocessing.get_default_pipeline()
from texthero import nlp
string = pd.Series('Donald Trump said he was "medication free" and revealed more details of his fight with Covid-19 in a televised interview aired Friday, one week after he was hospitalized with the virus.')
nlp.named_entities(string)[0]
nlp.noun_chunks(string)[0]
from texthero import representation, visualization
df['pca'] = (

    df['text']

    .pipe(preprocessing.clean)

    .pipe(representation.tfidf, max_features=1000)

    .pipe(representation.pca)

)



visualization.scatterplot(df, 'pca', color='author', title="Spooky Author Identification")
df['tfidf'] = (df['text'].pipe(preprocessing.clean).pipe(representation.tfidf, max_features=1000))



df['labels'] = (df['tfidf'].pipe(representation.kmeans, n_clusters=3).astype(str))



df['pca'] = df['tfidf'].pipe(representation.pca)



visualization.scatterplot(df, 'pca', color='labels', title="Spooky Author Identification using K-means")
visualization.wordcloud(df['clean_text'])
visualization.top_words(df['clean_text']).head(5)