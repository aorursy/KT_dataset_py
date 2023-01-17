import pandas as pd

dados = pd.read_csv('/kaggle/input/imdb-ptbr/imdb-reviews-pt-br.csv')

dados.head()
classificacao = dados['sentiment'].replace(['neg','pos'],[0,1])
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression



regressao_logistica = LogisticRegression(solver = "lbfgs")



tfidf = TfidfVectorizer(lowercase=False, max_features=50)

tfidf_bruto = tfidf.fit_transform(dados["text_pt"])

treino, teste, classe_treino, classe_teste = train_test_split(tfidf_bruto,

                                                              classificacao,

                                                              random_state = 42)

regressao_logistica.fit(treino, classe_treino)

acuracia_tfidf_bruto = regressao_logistica.score(teste, classe_teste)

print(acuracia_tfidf_bruto)

from string import punctuation

from unidecode import unidecode

import nltk

from nltk import tokenize, FreqDist, ngrams



todas_palavras = ' '.join([texto for texto in dados["text_pt"]])

stemmer = nltk.RSLPStemmer()



# Remover as palavras de cada linha do data frame

pontuacao = [ponto for ponto in punctuation]

palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")

# adicionando pontuacao na lista de stopwords

stopwords = palavras_irrelevantes + pontuacao + ['...']

token_pontuacao = nltk.WordPunctTokenizer()

# Limpando dados desnecessarios

coluna_filtrada= list()

for opiniao in dados.text_pt:

    nova_frase = list()

    palavras_texto = token_pontuacao.tokenize(opiniao.lower())

    for palavra in palavras_texto:

        if palavra not in stopwords: # stopwords

            nova_frase.append(stemmer.stem(unidecode(palavra)))

    coluna_filtrada.append(' '.join(nova_frase))



# Sobrescrevendo a coluna text_pt

dados.text_pt = coluna_filtrada
tfidf = TfidfVectorizer(lowercase=False, ngram_range = (1,2))

vetor_tfidf = tfidf.fit_transform(dados["text_pt"])

treino, teste, classe_treino, classe_teste = train_test_split(vetor_tfidf,

                                                              classificacao,

                                                              random_state = 42)

regressao_logistica.fit(treino, classe_treino)

acuracia_tfidf_ngrams = regressao_logistica.score(teste, classe_teste)

print(acuracia_tfidf_ngrams)
