import re

import nltk

import spacy



import pandas as pd



import matplotlib.pyplot as plt



from wordcloud import WordCloud

from unidecode import unidecode

from string import punctuation

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from nltk import word_tokenize

from nltk.corpus import stopwords



from spacy.lang.pt.stop_words import STOP_WORDS





%matplotlib inline
def limpar_texto(text):

    

    # Colocando todas as letras do texto em caixa baixa:

    text = text.lower()

    # Excluindo citações com @:

    text = re.sub('@[^\s]+', '', text)

    # Excluindo acentuação das palavras:

    text = unidecode(text)

    # Excluindo html tags, como <strong></strong>:

    text = re.sub('<[^<]+?>','', text)

    # Excluindo os números:

    text = ''.join(c for c in text if not c.isdigit())

    # Excluindo URL's:

    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)

    # Excluindo pontuação:

    text = ''.join(c for c in text if c not in punctuation)

    

    # Retornando o texto tratado tokenizado:

    

    return word_tokenize(text)
# O texto abaixo contém todas as situações para que seja feita a limpeza:



texto = """

<strong>Olá</strong> @usuario, vamos testar a função #clean_text?

Caso tenha dúvidas, uma boa pesquisa no www.google.com pode ajudar!

Mesmo que você tenha que pesquisar 100 vezes!

"""



texto_limpo = limpar_texto(texto)

print(texto_limpo)
# Removendo as stopwords utilizando a lista do nltk e do spacy:



sw = list(set(stopwords.words('portuguese') + list(STOP_WORDS)))



def remove_stop_words(texts, stopwords = sw):

      

    new_texts = list()

    

    for word in texts:

        if word not in stopwords:

            new_texts.append(''.join(word))



    return new_texts

print(sw)
texto_sem_stop_words = remove_stop_words(texto_limpo)

print(texto_sem_stop_words)
texto_sem_stop_words = remove_stop_words(texto_limpo, sw + ['cleantext'])

print(texto_sem_stop_words)
# primeiramente é necessário realizar a instalação abaixo:



!python -m spacy download pt
# Vamos criar uma função que mostra o texto original, a interpretação - da função - semântica dela e o lema



nlp = spacy.load("pt")



def verificar_lemma(words):

    

    text = ""

    pos = ""

    lemma = ""

    for word in nlp(words):

        text += word.text + "\t"

        pos += word.pos_ + "\t"

        lemma += word.lemma_ + "\t"



    print(text)

    print(pos)

    print(lemma)
verificar_lemma('o sentido desta frase está errado')
verificar_lemma('você está se sentindo bem?')
# Vamos criar uma função que mostra o texto original e o stem de cada palavra



def verificar_radical(words):

    

    stemmer = nltk.stem.SnowballStemmer('portuguese')

    text = ""

    stem = ""

    

    for word in word_tokenize(words):



        text += word + "\t"

        stem += stemmer.stem(word) + "\t"

    

    print(text)

    print(stem)
verificar_radical('o sentido desta frase esta errado')
verificar_radical('você está se sentindo bem?')
def nuvem_palavras(textos):

    

    # Juntando todos os textos na mesma string

    todas_palavras = ' '.join([texto for texto in textos])

    # Gerando a nuvem de palavras

    nuvem_palvras = WordCloud(width= 800, height= 500,

                              max_font_size = 110,

                              collocations = False).generate(todas_palavras)

    # Plotando nuvem de palavras

    plt.figure(figsize=(24,12))

    plt.imshow(nuvem_palvras, interpolation='bilinear')

    plt.axis("off")

    plt.show()
df = pd.read_csv('../input/imdb-ptbr/imdb-reviews-pt-br.csv', nrows=1000)
# vamos ver as primeiras cinco linhas do dataset:

df.head(5)
# Construindo a nuvem de palavras:

nuvem_palavras(df["text_pt"])
def countvectorizer(textos):



    vect = CountVectorizer()

    text_vect = vect.fit_transform(textos)

    

    return text_vect



def tfidfvectorizer(textos):

    

    vect = TfidfVectorizer(max_features=50)

    text_vect = vect.fit_transform(textos)

    

    return text_vect
class preprocess_nlp(object):

    

    def __init__(self, texts, stopwords = True, lemma=False, stem=False, wordcloud=True, numeric='tfidf'):

        

        self.texts = texts

        self.stopwords = stopwords

        self.lemma = lemma

        self.stem = stem

        self.wordcloud = wordcloud

        self.numeric = numeric

        self.new_texts = None

        self.stopwords_list = list()

        

    def clean_text(self):



        new_texts = list()



        for text in self.texts:



            text = text.lower()

            text = re.sub('@[^\s]+', '', text)

            text = unidecode(text)

            text = re.sub('<[^<]+?>','', text)

            text = ''.join(c for c in text if not c.isdigit())

            text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))', '', text)

            text = ''.join(c for c in text if c not in punctuation)

            new_texts.append(text)

        

        self.new_texts = new_texts



    def create_stopwords(self):

        

        stop_words = list(set(stopwords.words('portuguese') + list(STOP_WORDS)))

        

        for word in stop_words:



            self.stopwords_list.append(unidecode(word))

       

    

    def add_stopword(self, word):

        

        self.stopwords_list += [word]

        



    def remove_stopwords(self):



        new_texts = list()



        for text in self.new_texts:



            new_text = ''



            for word in word_tokenize(text):



                if word.lower() not in self.stopwords_list:



                    new_text += ' ' + word



            new_texts.append(new_text)



        self.new_texts = new_texts





    def extract_lemma(self):

        

        nlp = spacy.load("pt")

        new_texts = list()



        for text in self.texts:



            new_text = ''



            for word in nlp(text):



                new_text += ' ' + word.lemma_



            new_texts.append(new_text)

        

        self.new_texts = new_texts

    



    def extract_stem(self):



        stemmer = nltk.stem.SnowballStemmer('portuguese')

        new_texts = list()



        for text in self.texts:



            new_text = ''



            for word in word_tokenize(text):



                new_text += ' ' + stemmer.stem(word)



            new_texts.append(new_text)



        self.new_texts = new_texts

    



    def word_cloud(self):



        all_words = ' '.join([text for text in self.new_texts])

        word_cloud = WordCloud(width= 800, height= 500,

                               max_font_size = 110,

                               collocations = False).generate(all_words)

        plt.figure(figsize=(24,12))

        plt.imshow(word_cloud, interpolation='bilinear')

        plt.axis("off")

        plt.show()

        



    def countvectorizer(self):



        vect = CountVectorizer()

        text_vect = vect.fit_transform(self.new_texts)



        return text_vect

    



    def tfidfvectorizer(self):



        vect = TfidfVectorizer(max_features=50)

        text_vect = vect.fit_transform(self.new_texts)



        return text_vect

    

    

    def preprocess(self):



        self.clean_text()

        

        if self.stopwords == True:

            self.create_stopwords()

            self.remove_stopwords()

            

        if self.lemma == True:

            self.extract_lemma()

        

        if self.stem == True:

            self.extract_stem() 

        

        if self.wordcloud == True:

            self.word_cloud()

        

        if self.numeric == 'tfidf':

            text_vect = self.tfidfvectorizer()

        elif self.numeric == 'count':

            text_vect = self.countvectorizer()

        else:

            print('metodo nao mapeado!')

            exit()

            

        return text_vect
prepro = preprocess_nlp(df['text_pt'], numeric='count')

#adicionando as palavras filme e filmes na lista de palavras de parada, pois elas são irrelevantes neste contexto

prepro.add_stopword('filme') 

prepro.add_stopword('filmes') 

sparse_matrix = prepro.preprocess()