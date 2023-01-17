# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import confusion_matrix, f1_score

import requests



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

import nltk

from unidecode import unidecode

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from sklearn.metrics import classification_report, confusion_matrix
# Carregar datasets



sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

test = pd.read_csv("../input/nlp-getting-started/test.csv", index_col='id')

train = pd.read_csv("../input/nlp-getting-started/train.csv", index_col='id')
# Retirar pontuação



for col in train.columns:

    if not train[col].dtype == 'int':

        train[col] = train[col].str.lower().replace(r'[^\w\s]','',regex=True)



# Retirar números



train['keyword'] = train.keyword.str.replace(r'\d','',regex=True)

train['location'] = train.location.str.replace(r'\d','',regex=True)



# Remover acentos

train['text'] = train.text.apply(unidecode)



# Remover stopwords



def remover_stopwords(frase):

    

    stop_words = set(stopwords.words('english')) 

    tokens = word_tokenize(frase) 

    frase_filtrada = [p for p in tokens if not p in stop_words] 

    return ' '.join(frase_filtrada)



train['text'] = train.text.apply(remover_stopwords)



lemmatizer = WordNetLemmatizer()



train = train.dropna(subset=['keyword'])

train.keyword = train.keyword.apply(lemmatizer.lemmatize)

train.location.fillna('unknown',inplace=True)
X = train.drop(columns='target')

y = train.target



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.4)
## Criamos uma pipeline para classificação



vectorizer = CountVectorizer()

classifier = RandomForestClassifier()



pipe = Pipeline([('vectorizer',vectorizer),

                ('clf',classifier)])



pipe.fit(X_train.text, y_train)
## Invocamos a pipeline com as métricas escolhidas, classification report e matriz de confusão



y_pred = pipe.predict(X_val.text)



print(classification_report(y_val, y_pred), confusion_matrix(y_val, y_pred))
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering

from sklearn.preprocessing import StandardScaler, FunctionTransformer

from sklearn.feature_extraction.text import HashingVectorizer

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# Carregar dataset



airbnb = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv", index_col='id')



airbnb.head()
## !Utilizar get dummies nos bairros e no tipo de quarto



## Remover colunas de medidas que não são de nosso interesse, como o nome do anfitrião e os dados sobre críticas.



to_drop = ['host_id','host_name','number_of_reviews','last_review','reviews_per_month','calculated_host_listings_count']



airbnb = airbnb.drop(columns=to_drop)



## Também adequamos o tipo de algumas variáveis



airbnb[['neighbourhood_group','neighbourhood','room_type']] = airbnb[['neighbourhood_group','neighbourhood','room_type']].astype('category')



airbnb.info()
get_categorical = FunctionTransformer(lambda x: pd.get_dummies(x[['neighbourhood_group','neighbourhood','room_type']], drop_first=True))



get_numerical = FunctionTransformer(lambda x:x[['latitude','longitude','price','minimum_nights','availability_365']])

                                      

scaler = StandardScaler()



get_text = FunctionTransformer(lambda x: x['name'])

                                      

vectorizer = HashingVectorizer()



tratar_e_combinar = FeatureUnion(transformer_list=

                    [('categoricas',get_categorical),

                    ('numericas',Pipeline([

                        ('retirar',get_numerical),

                        ('scaler',scaler)])),

                     ('texto',Pipeline([

                         ('retirar',get_text),

                         ('vectorizer',vectorizer)

                     ]))])



kmeans = KMeans(n_clusters=5, random_state=2)



pipe = Pipeline([('tratamento',tratar_e_combinar),

                 ('modelo',kmeans)])



X_train, X_test = train_test_split(airbnb, test_size=0.8)
### O algoritmo não funcionou :(. Acredito que a vetorização do texto tenha ficado pesada pela falta de tratamento.



X = tratar_e_combinar.fit_transform(X_train)

SpectralClustering().fit(X.toarray())



## 
# Carregar dataset



OptimalPolicy_angletol45 = pd.read_csv("../input/-reinforcement-learning-from-scratch-in-python/OptimalPolicy_angletol45.csv")
OptimalPolicy_angletol45.info()