# Importando as bibliotecas necessárias:
%matplotlib inline
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import gensim
from gensim.models import LdaModel
from gensim import models, corpora, similarities
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import RSLPStemmer
import time
from nltk import FreqDist
from scipy.stats import entropy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import os # accessing directory structure
from wordcloud import WordCloud, STOPWORDS

print(os.listdir('../input'))
# boatos.csv tem 1170 linhas
df = pd.read_csv('../input/boatos.csv', delimiter=',')
df.dataframeName = 'boatos.csv'
nRow, nCol = df.shape
print(f'Há {nRow} linhas e {nCol} colunas')
df.head(5)
df.info()
def initial_clean(text):
    """
    Função para limpeza de textos de websites, emails e pontuação
    Também converte para minúsculas
    """
    text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
    text = re.sub("[^a-zA-Z ]", "", text)
    text = text.lower() # lower case the text
    text = nltk.word_tokenize(text)
    return text

stop_words = stopwords.words('portuguese') #stop words in Português
def remove_stop_words(text):
    """
    Função apra remover "stop-words"
    """
    return [word for word in text if word not in stop_words]

stemmer = nltk.stem.RSLPStemmer() # Stemmer in Portuguese
def stem_words(text):
    """
    Função para iguralar singular e plural
    """
    try:
        text = [stemmer.stem(word) for word in text]
        text = [word for word in text if len(word) > 1] # make sure we have no 1 letter words
    except IndexError: # the word "oed" broke this, so needed try except
        pass
    return text

def apply_all(text):
    """
    Aglutina todas as funções anteriores
    """
    return stem_words(remove_stop_words(initial_clean(text)))
# clean text and title and create new column "tokenized"
t1 = time.time()
df['tokenized'] = df['hoax'].apply(apply_all)
t2 = time.time()
print("Time to clean and tokenize", len(df), "articles:", (t2-t1)/60, "min")
df.head()
# first get a list of all words
all_words = [word for item in list(df['tokenized']) for word in item]
# use nltk fdist to get a frequency distribution of all words
fdist = FreqDist(all_words)
len(fdist) # number of unique words
top= fdist.most_common()
print(top)

k = 50000
top_k_words = fdist.most_common(k)
top_k_words[-10:]
# choose k and visually inspect the bottom 10 words of the top k
# Escolhe o valor de k e veririca visualmente as 10 palavras menos usadas das k mais frequentes
k = 1500
top_k_words = fdist.most_common(k)
top_k_words[-10:]
# definição de uma função para encontrar as palavras mais frequentes
# fdist.most_common(k) monta um vetor de tuplas (palavra_1,quantidade_1), (palavra_2,quantidade_2), ...
# *fdist.most_common(k) separa os itens (tuplas) do vetor em argumentos para a função zip
# a função zip agrega todas as palavras de cada tupla em uma nova tupla, e todos os numeros (quantidades) em outra tupla
# Ex: (palavra_1, plavra_2, ...), (4,7,...)
# top_k_words pega primeira tupla, a tupla das palavras
top_k_words,_ = zip(*fdist.most_common(k))
# a função set torna a tupla em um conjunto de valores que não tem indices, nem pode ter itens repetidos.
# a ordem é aleatoria e não importa
top_k_words = set(top_k_words)
def keep_top_k_words(text):
    return [word for word in text if word in top_k_words]
# aplica a função à coluna 'tokenized' para manter só as mais frequentes (palavras incomuns são removidas)
df['tokenized'] = df['tokenized'].apply(keep_top_k_words)
# O tamanho (coluna 'doc_len') é calculado
df['doc_len'] = df['tokenized'].apply(lambda x: len(x))
# coloca os tamnahos em um vetor
doc_lengths = list(df['doc_len'])
# remove a coluna 'doc_len' do dataframe
df.drop(labels='doc_len', axis=1, inplace=True)

print("length of list:",len(doc_lengths),
      "\naverage document length", np.average(doc_lengths),
      "\nminimum document length", min(doc_lengths),
      "\nmaximum document length", max(doc_lengths))
# plot a histogram of document length
num_bins = 1000
fig, ax = plt.subplots(figsize=(20,6));
# the histogram of the data
n, bins, patches = ax.hist(doc_lengths, num_bins)
ax.set_xlabel('Document Length (tokens)', fontsize=15)
ax.set_ylabel('Normed Frequency', fontsize=15)
ax.grid()
ax.set_xticks(np.logspace(start=np.log10(50),stop=np.log10(2000),num=8, base=10.0))
plt.xlim(0,2000)
ax.plot([np.average(doc_lengths) for i in np.linspace(0.0,0.0035,100)], np.linspace(0.0,0.0035,100), '-',
        label='average doc length')
ax.legend()
ax.grid()
fig.tight_layout()
plt.show()
# MANTEM APENAS OS ARTIGOS COM MAIS DE 40 TOKENS
df = df[df['tokenized'].map(len) >= 40]

# make sure all tokenized items are lists
df = df[df['tokenized'].map(type) == list]
df.reset_index(drop=True,inplace=True)
print("Após a limpeza e exclusão de artigos curtos, o dataframe tem agora:", len(df), "artigos")
df.head()
# Cria uma lista aleatoria de itens TRUE or FALSE, onde 98% é TRUE, para ser utilizado na divisão do dataframe
msk = np.random.rand(len(df)) < 0.98
#Divisão do dataframe usando a lista de valores TRUE e FALSE
# para treinamento os valores TRUE
train_df = df[msk]
train_df.reset_index(drop=True,inplace=True)
# para teste são usados os valores FALSE
test_df = df[~msk]
test_df.reset_index(drop=True,inplace=True)
train_df.to_csv('train_df.csv')
test_df.to_csv('test_df.csv')
# Tamanho dos conjutnos de ddos de treino e teste
print(len(df),len(train_df),len(test_df))
def train_lda(data):
    """
    Esta função treina o modelo LDA
    Configuramos os parametros como o número de tópicos, o 'chunksize' para usar o método de Hoffman
    Fazemos duas passagens de dados já que o dataset é pequeno, queremos que o as distribuições se estabilizem
    """
    num_topics = 100
    chunksize = 300
    dictionary = corpora.Dictionary(data['tokenized'])
    corpus = [dictionary.doc2bow(doc) for doc in data['tokenized']]
    t1 = time.time()
    # low alpha means each document is only represented by a small number of topics, and vice versa
    # low eta means each topic is only represented by a small number of words, and vice versa
    lda = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary,
                   alpha=1e-2, eta=0.5e-2, chunksize=chunksize, minimum_probability=0.0, passes=2)
    t2 = time.time()
    print("Tempo para treinar o modelo LDA com ", len(df), "artigos: ", (t2-t1)/60, "min")
    return dictionary,corpus,lda
dictionary,corpus,lda = train_lda(train_df)
# O método show_topics mostra as palavras mais frequentes (quantidade definida por 'num_words') na quantidade 'num_topics' de tópicos aleatórios.
lda.show_topics(num_topics=5, num_words=50)
# mostra um tópico especifico. argumentos: id do topico e quantidade de palavras (mais significativas)
lda.show_topic(topicid=98, topn=2)
# seleciona um artigo aleatoriamente de train_df
random_article_index = np.random.randint(len(train_df)) # pega um numero aleatorio menor que o tamanho de train_df
bow = dictionary.doc2bow(train_df.iloc[random_article_index,4]) # Lista de tuplas do artigo com (token_id, token_count)
print(random_article_index)
print(train_df.iloc[random_article_index,4])
# Lista os topicos do artigo escolhido aleatoriamente (primeiro item das tuplas)
doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=bow)]) # a função get_document_topics retorna a distribuição dos topicos
# Gráfico da distribuição de topicos do artigo
fig, ax = plt.subplots(figsize=(12,6));
# Histograma dos dados
patches = ax.bar(np.arange(len(doc_distribution)), doc_distribution)
ax.set_xlabel('ID do tópico', fontsize=15)
ax.set_ylabel('Contribuição do tópico', fontsize=15)
ax.set_title("Distribuição de tópicos do artigo " + str(random_article_index), fontsize=20)
#ax.set_xticks(np.linspace(10,100,10))
#fig.tight_layout()
plt.show()
# Os 5 tópicos que mais contribuem e suas palavras
for i in doc_distribution.argsort()[-5:][::-1]:
    print(i, lda.show_topic(topicid=i, topn=10), "\n")
# Escolha de um artigo aleatório dos dados de teste (test_df)
random_article_index = np.random.randint(len(test_df))
print(random_article_index)
new_bow = dictionary.doc2bow(test_df.iloc[random_article_index,4])
print(test_df.iloc[random_article_index,1])
new_doc_distribution = np.array([tup[1] for tup in lda.get_document_topics(bow=new_bow)])
# Gráfico da distribuição de topicos do artigo
fig, ax = plt.subplots(figsize=(12,6));
# Histograma dos dados
patches = ax.bar(np.arange(len(new_doc_distribution)), new_doc_distribution)
ax.set_xlabel('ID do tópico', fontsize=15)
ax.set_ylabel('Contribuição do tópico', fontsize=15)
ax.set_title("Distribuição dos tópicos para um artigo não visto", fontsize=20)
ax.set_xticks(np.linspace(10,100,10))
fig.tight_layout()
plt.show()
# Os 5 tópicos que mais contribuem e suas palavras
for i in new_doc_distribution.argsort()[-5:][::-1]:
    print(i, lda.show_topic(topicid=i, topn=10), "\n")


