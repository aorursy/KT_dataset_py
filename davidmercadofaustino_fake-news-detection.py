# Importação de bibliotecas para utilização na Leitura e na manipulação de dados

import pandas as pd

import numpy as np
# Carregamento pelo pandas por meio do read CSV

df_fake_news = pd.read_csv("../input/fake-news-detection/data.csv")
df_fake_news.head(5) 
# Analise da qualidade dos dados importados

df_fake_news.describe()
#Importação do Collection para observar melhor os dados que serão analisados

from collections import Counter
Counter(df_fake_news["Label"])
# Alterar os dados da coluna Label para melhor vizualização dos resultados

df_fake_news['Label'].replace(1, 'News',regex=True, inplace=True) 
df_fake_news['Label'].replace(0, 'Fake News',regex=True, inplace=True)
Counter(df_fake_news["Label"])
#importação do NLTK para ajudar no processamento das informações para o Naive Bayes

import nltk

nltk.download('stopwords')
# Importação das stopwords para fazer a limpeza da coluna Headline

from nltk.corpus import stopwords
# Estabelecendo as stopwords em inglês

list_stops_words = stopwords.words("english")
df_fake_news["Headline"] = df_fake_news['Headline'].str.lower()
df_fake_news["Headline"].replace('\n',' ', regex=True, inplace=True)
#importação do sklearn para criar modelo de previsão

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(df_fake_news["Headline"], df_fake_news["Label"], 

                                                    test_size=0.30, 

                                                    random_state=42)
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer
classify = Pipeline(

                [('vect', CountVectorizer(stop_words= list_stops_words)), 

                 ('tfidf', TfidfTransformer()),

                 ('clf', MultinomialNB())

                 ])
# treinamento do nosso modelo

classify.fit(X_train, y_train)
# Importação do metrics para medir o aprendizado da maquina

from sklearn import metrics
# Medição da precisão do modelo utilizando as amostras

classify.score(X_test, y_test)
# Capacidade de predição do modelo

preds = classify.predict(X_test)

print(metrics.classification_report(y_test, preds))
texto  = "Trump and his allies respond with pseudo-science as US death toll hits 150,000"
# Testando a predição do modelo

classify.predict([texto])
# Base da decisão pela classificação como News

classify.predict_proba([texto])