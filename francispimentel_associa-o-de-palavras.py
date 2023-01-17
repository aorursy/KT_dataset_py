!pip install apyori
import numpy as np

import pandas as pd

import string

from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

from apyori import apriori

from nltk.corpus import stopwords

from nltk import pos_tag
df = pd.read_table('../input/SW_EpisodeIV.txt',delim_whitespace=True, header=0, escapechar='\\')

len(df)
all_dialogues = list(df.dialogue.values)

print('Tamanho: ', len(all_dialogues))

print(all_dialogues[:10])
all_sents = [[w.lower() for w in word_tokenize(sen) if not w in string.punctuation]\

            for sen in all_dialogues]



x = []

y = []



print(all_sents[:10])

print(len(all_sents))
stopwords_list = stopwords.words('english')



def remover_pontuacao_e_stopwords(doc):

    non_stopwords = [w for w in doc if not w[0].lower() in stopwords_list]

    non_punctuation = [w for w in non_stopwords if not w[0] in string.punctuation]

    return non_punctuation



def remover_apenas_pontuacao(doc):

    non_punctuation = [w for w in doc if not w[0] in string.punctuation]

    return non_punctuation
print('Frase original: ', all_sents[0])

print('Frase tratada: ', remover_pontuacao_e_stopwords(all_sents[0]))
sentencas_sem_stopwords = [remover_pontuacao_e_stopwords(s) for s in all_sents]

sentencas_com_stopwords = [remover_apenas_pontuacao(s) for s in all_sents]
rules_sem_stopwords = apriori(sentencas_sem_stopwords, min_support=0.008, min_confidence=0.3, min_lift=2)

association_results = list(rules_sem_stopwords)
print('Sem stopwords:')

for result in association_results[:15]:

    print('--------')

    print(result)
rules_com_stopwords = apriori(sentencas_com_stopwords, min_support=0.008, min_confidence=0.3, min_lift=2)

association_results = list(rules_com_stopwords)
print('Sem stopwords:')

for result in association_results[:15]:

    print('--------')

    print(result)