import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

nltk.download("stopwords")

nltk.download("punkt")

from pprint import pprint
stopwordPortugues = nltk.corpus.stopwords.words("portuguese")

print(np.transpose(stopwordPortugues))
sample_text = """O menino gosta de jogar futebol aos finais de semana.

Ele gosta de jogar com seus amigos Marcos e João, mas não gosta de brincar

com a irmã Marcela

"""

tokenizacao_sentencas = nltk.sent_tokenize

sample_sentence = tokenizacao_sentencas(text=sample_text)

pprint(sample_sentence)
len(sample_sentence)
sample_sentence = "O menino gosta de jogar futebol aos finais de semana"

tokenizacao_palavras = nltk.word_tokenize

sample_words = tokenizacao_palavras(sample_sentence)

pprint(sample_words)
len(sample_sentence)
from nltk.stem import PorterStemmer

from nltk.stem import RSLPStemmer

nltk.download("rslp")
ps = PorterStemmer()

stemmer = RSLPStemmer()



print(ps.stem("jumping"))

print(stemmer.stem("amoroso"))

print(stemmer.stem("amorosa"))

print(stemmer.stem("amados"))
from nltk.stem import SnowballStemmer #better for pt-br 



print("linguagens suportadas %s", SnowballStemmer.languages)
ss = SnowballStemmer("portuguese")

print(ss.stem("casado"))

print(ss.stem("casarão"))

print(ss.stem("casa"))
#Bag of Words

sentenca = "O IGTI oferece especializacao em Deep Learning. Deep Learning e utilizado em diversas aplicacoes. As aplicacoes de deep learning sao estudadas nesta especializacao. O IGTI tambem oferece diversos bootcamps"
sentenca = sentenca.lower()
print(sentenca)
tokenizacao_sentencas = nltk.sent_tokenize

samples_sentence = tokenizacao_sentencas(text = sentenca)

pprint(samples_sentence)
samples_sentence[0]
tokenizacao_palavras = nltk.word_tokenize

list_words = []

for i in range(len(samples_sentence)):

    sample_words = tokenizacao_palavras(text = samples_sentence[i])

    list_words.extend(sample_words)
print(list_words)
def tokenizaPalavras(sentenca):

    sample_words = tokenizacao_palavras(text = sentenca)

    return sample_words



def removeStopWords(list_of_words):

    my_stop_words = ["o", "em", "as", "de", "sao", "nesta", ".", "e", "a", "na", "do"]

    list_cleaned = set(list_of_words) - set(my_stop_words)

    return list_cleaned



my_BoW = removeStopWords(list_words)
print(my_BoW)
def bagofwords(sentence, words):

    sentence_words = tokenizaPalavras(sentence)

    bag = np.zeros(len(words))

    for sw in sentence_words:

        for i, word in enumerate(words):

            if word == sw:

                bag[i] += 1

                

    return np.array(bag)



sentenca_teste = "o igti oferece especializacao em deep learning e o igti oferece bootcamp"

print(bagofwords(sentenca_teste, my_BoW))