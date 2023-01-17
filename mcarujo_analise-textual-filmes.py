import pandas as pd

dados = pd.read_csv('/kaggle/input/imdb-ptbr/imdb-reviews-pt-br.csv')

dados.head()
# 'Traduzir' a coluna sentiment para números

classificacao = dados['sentiment'].replace(['neg','pos'],[0,1])



# Importandando o CountVectorizer para criação do vocabulário e saco de palavras (bag of words)

from sklearn.feature_extraction.text import CountVectorizer

modelo_vetorizador = CountVectorizer()



# gerando meu bag of words

saco_de_palavras = modelo_vetorizador.fit_transform(dados['text_pt'])



# Salvando o vocabulario, posteriormente será nossas colunas do dataframe

vocabulario = modelo_vetorizador.get_feature_names()



# Gerando um dataframe com o saco de palavras e vocabulario

dados_manipulado = pd.SparseDataFrame(saco_de_palavras, columns=vocabulario)



# Nosso vocabulario tem 129 622 palavras

print("Saco de Palavras dimensoes",dados_manipulado.shape)

print("Vocabulario dimensoes",len(vocabulario))

print("Classificacao",classificacao.shape)
# Dividindo o conjunto de dados para treinamento

from sklearn.model_selection import train_test_split

treino, teste, classe_treino, classe_teste = train_test_split(saco_de_palavras, classificacao,random_state = 40)
# Treinamento e print do resultado da validacao

from sklearn.linear_model import LogisticRegression

modelo = LogisticRegression()

modelo.fit(treino,classe_treino)

acuracia = modelo.score(teste, classe_teste)

print("Acuracia : ",acuracia)
%matplotlib inline

from wordcloud import WordCloud

import matplotlib.pyplot as plt



def nuvem_palavras(texto, coluna_texto,sentimento):

    # Separar nuvem por sentimento

    texto = texto.query(f"sentiment == '{sentimento}'")

    # Juntando todos os textos na mesma string

    todas_palavras = ' '.join([texto for texto in texto[coluna_texto]])

    # Gerando a nuvem de palavras

    nuvem_palvras = WordCloud(width= 800, height= 500,

                              max_font_size = 110,

                              collocations = False).generate(todas_palavras)

    # Plotando nuvem de palavras

    plt.figure(figsize=(24,12))

    plt.imshow(nuvem_palvras, interpolation='bilinear')

    plt.axis("off")

    plt.show()
nuvem_palavras(dados, "text_pt", "pos")
nuvem_palavras(dados, "text_pt", "neg")
import nltk

from nltk import tokenize, FreqDist

todas_palavras = ' '.join([texto for texto in dados["text_pt"]])
# Tokerizar para verificar a frequencia de cada palavra

token_espaco = tokenize.WhitespaceTokenizer()

token_frase = token_espaco.tokenize(todas_palavras)

frequencia = nltk.FreqDist(token_frase)

df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()),

                              "Frequência": list(frequencia.values())})

# 5 Palavras mais comunas na base de dados

df_frequencia.sort_values(by='Frequência', inplace=True, ascending=False)

df_frequencia.head()
import seaborn as sns



# Plotar as palavras mais comuns da base de dados

plt.figure(figsize=(12,8))

ax = sns.barplot(data = df_frequencia.head(15), x = "Palavra", y = "Frequência", color = 'gray')

ax.set(ylabel = "Contagem")

plt.show()

# As palavras mais frequentes não nos desmonstram nenhum sentimento sobre a avaliação...
# Remover as palavras de cada linha do data frame

palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")



# Limpando dados desnecessarios

coluna_filtrada= list()

for opiniao in dados.text_pt:

    nova_frase = list()

    palavras_texto = opiniao.split()

    for palavra in palavras_texto:

        if palavra not in palavras_irrelevantes:

            nova_frase.append(palavra)

    coluna_filtrada.append(' '.join(nova_frase))



# Sobrescrevendo a coluna text_pt

dados.text_pt = coluna_filtrada
def nuvem_palavras_filtradas(texto, coluna_texto,sentimento):

    # Separar nuvem por sentimento

    texto = texto.query(f"sentiment == '{sentimento}'")

    # Juntando todos os textos na mesma string

    todas_palavras = ' '.join([texto for texto in texto[coluna_texto]])

    # Gerando a nuvem de palavras

    nuvem_palvras = WordCloud(width= 800, height= 500,

                              max_font_size = 110,

                              collocations = False).generate(todas_palavras)

    # Plotando nuvem de palavras

    plt.figure(figsize=(24,12))

    plt.imshow(nuvem_palvras, interpolation='bilinear')

    plt.axis("off")

    plt.show()
nuvem_palavras_filtradas(dados, "text_pt", "pos")
nuvem_palavras_filtradas(dados, "text_pt", "neg")
# gerando meu bag of words

saco_de_palavras = modelo_vetorizador.fit_transform(dados['text_pt'])



# Salvando o vocabulario, posteriormente será nossas colunas do dataframe

vocabulario = modelo_vetorizador.get_feature_names()



# Gerando um dataframe com o saco de palavras e vocabulario

dados_manipulado = pd.SparseDataFrame(saco_de_palavras, columns=vocabulario)



# Nosso vocabulario tem 129 612 palavras

print("Saco de Palavras dimensoes",dados_manipulado.shape)

print("Vocabulario dimensoes",len(vocabulario))

print("Classificacao",classificacao.shape)
# Dividindo o conjunto de dados para treinamento

treino, teste, classe_treino, classe_teste = train_test_split(saco_de_palavras, classificacao,random_state = 40)
# Exibindo dimensão dos conjuntos de treino e de validacao

print("Conjunto de Treino: ",treino.shape, " - Conjunto de Validacao: ", teste.shape)

print("Saida Desejada de Treino: ",classe_treino.shape, " - Sai Desejada de Validacao: ", classe_teste.shape)

# Treinamento e print do resultado da validacao

modelo = LogisticRegression()

modelo.fit(treino,classe_treino)

acuracia = modelo.score(teste, classe_teste)

print("Acuracia : ",acuracia)
from string import punctuation

from unidecode import unidecode

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
todas_palavras = ' '.join([texto for texto in dados["text_pt"]])

# Tokerizar para verificar a frequencia de cada palavra agora que excluimos a pontuacao

token_espaco = tokenize.WhitespaceTokenizer()

token_frase = token_espaco.tokenize(todas_palavras)

frequencia = nltk.FreqDist(token_frase)

df_frequencia = pd.DataFrame({"Palavra": list(frequencia.keys()),

                              "Frequência": list(frequencia.values())})



# 5 Palavras mais comunas na base de dados

df_frequencia.sort_values(by='Frequência', inplace=True, ascending=False)

df_frequencia.head()
# Plotar as palavras mais comuns da base de dados

plt.figure(figsize=(12,8))

ax = sns.barplot(data = df_frequencia.head(15), x = "Palavra", y = "Frequência", color = 'gray')

ax.set(ylabel = "Contagem")

plt.show()

# As palavras mais frequentes não nos desmonstram nenhum sentimento sobre a avaliação...
modelo_vetorizador = CountVectorizer()



# gerando meu bag of words

saco_de_palavras = modelo_vetorizador.fit_transform(dados['text_pt'])



# Salvando o vocabulario, posteriormente será nossas colunas do dataframe

vocabulario = modelo_vetorizador.get_feature_names()



# Gerando um dataframe com o saco de palavras e vocabulario

dados_manipulado = pd.SparseDataFrame(saco_de_palavras, columns=vocabulario)



# Nosso vocabulario tem 129 622 palavras

print("Saco de Palavras dimensoes",dados_manipulado.shape)

print("Vocabulario dimensoes",len(vocabulario))

print("Classificacao",classificacao.shape)
# Dividindo o conjunto de dados para treinamento

from sklearn.model_selection import train_test_split

treino, teste, classe_treino, classe_teste = train_test_split(saco_de_palavras, classificacao,random_state = 40)



# Treinamento e print do resultado da validacao

modelo = LogisticRegression()

modelo.fit(treino,classe_treino)

acuracia = modelo.score(teste, classe_teste)

print("Acuracia : ",acuracia)