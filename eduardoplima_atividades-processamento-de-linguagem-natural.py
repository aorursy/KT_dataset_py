

import numpy as np 

import pandas as pd 

import os, json, glob, string, itertools, re

import nltk, spacy



#import calendar, datetime



from nltk.tokenize import word_tokenize





from functools import reduce



files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))



arquivo_teste = files[0]

json_arquivo = json.load(open(arquivo_teste, 'r'))

files[:25]
!python -m spacy download pt_core_news_sm
#!pip install https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-2.1.0/pt_core_news_sm-2.1.0.tar.gz 
diarios_dir = '/mnt/femurn'

json_dir = '/mnt/staging/datasets/femurn/json2'

#os.chdir(diarios_dir)

#arquivos = glob.glob("publicado*pdf")



#arquivos_json = os.listdir(json_dir)

#datas_arquivos_json = list(map(lambda x: re.search("diario_femurn_.+_(\d{4})(\d{2})(\d{2})_.+.json", x).groups(), arquivos_json))



def ja_existe(ano, mes, dia):

    return (ano, mes, dia) in datas_arquivos_json



def coletar_json():

    for i,a in enumerate(arquivos):

        diario_json = {}

        codigo_diario, ano, mes, dia, hash_diario = re.search('publicado_(\d+)_(\d+)-(\d+)-(\d+)_(\w+).*.pdf', a).groups()

        if ja_existe(ano, mes, dia):

            print("Arquivo %s já processado" % a)

            continue

        print('Arquivo %s será processado' % a)

        diario_json['codigo'] = codigo_diario

        diario_json['ano'] = ano

        diario_json['mes'] = mes

        diario_json['dia'] = dia

        diario_json['hash'] = hash_diario

        diario_json['secoes'] = []

        arquivo_json = "diario_femurn_%s_%s%s%s_%s.json" % (codigo_diario, ano, mes, dia, hash_diario)

        raw = parser.from_file(a)

        try:

            txt_diario = raw['content']

        except:

            print("Erro em %s" % a)

            continue

        txt_diario = re.sub("Rio Grande do Norte , \d{2} de \w+ de \d{4}   •   Diário Oficial dos Municípios do Estado do Rio Grande do Norte   •    ANO .+ | Nº \d{4}\s*\n", '', txt_diario)

        txt_diario = re.sub("www.diariomunicipal.com.br/femurn\s+\d+\s*\n", '', txt_diario)

        secoes = re.split('ESTADO DO RIO GRANDE DO NORTE \n', txt_diario)

        for idx_s,s in enumerate(secoes):

            secao = {}

            split_secao = s.split('\n')

            secao['jurisdicionado'] = split_secao[0] if split_secao[0] else split_secao[1]

            secao['publicacoes'] = []

            pubs = re.split('Código Identificador:\w*', s)

            for p in pubs:

                secao['publicacoes'].append(p)

            diario_json['secoes'].append(secao)

        json.dump(diario_json, open(os.path.join(json_dir,arquivo_json), 'w'))

        clear_output()
pubs_tokenizadas = []

publicacoes = []

for f in files[:25]:

    json_diario = json.load(open(f, 'r'))

    p = pd.io.json.json_normalize(json_diario, ['secoes', ['publicacoes']]).values

    for pub in p:

        publicacoes.append(pub[0])

        pubs_tokenizadas.append(word_tokenize(pub[0]))

len(pubs_tokenizadas)
#pubs_tokenizadas[0]
palavras_com_stopwords = [palavra.lower() for palavra in reduce(lambda x,y: x+y, pubs_tokenizadas)]

len(palavras_com_stopwords)
from nltk import FreqDist



#O objeto "freq"

freq = FreqDist(palavras_com_stopwords)

freq
from wordcloud import WordCloud

import matplotlib.pyplot as plt



words = ' '.join(palavras_com_stopwords)



wordcloud = WordCloud(max_font_size=100,width = 1600, height = 600).generate(words)

plt.figure(figsize=(16,9))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
def remover_stopwords(palavras):

    meses = ['janeiro','fevereiro','março','abril','maio','junho','julho','agosto','setembro','outubro','novembro','dezembro']



    stop_words_tce = ['n°','nº','cód','diário','r','v.'] + meses



    stopwords = set(nltk.corpus.stopwords.words("portuguese") + list(string.punctuation) + stop_words_tce + ['...', '•', '–'])



    return list(filter(lambda palavra: palavra not in stopwords 

                       and not re.match("(R\$)?(\s)?\d+(,|.)*\d*(,|.)*\d*", palavra)

                       and not re.match("^\s*$", palavra), palavras))

# Expressão regular adicional para remover números diversos como anos e valores

palavras_sem_stopwords = remover_stopwords(palavras_com_stopwords)
freq = FreqDist(palavras_sem_stopwords)

freq.plot(20, cumulative=False)
words = ' '.join(palavras_sem_stopwords)



wordcloud = WordCloud(max_font_size=100,width = 1600, height = 600).generate(words)

plt.figure(figsize=(16,9))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
palavras_postag_sem_stopwords = nltk.pos_tag(palavras_sem_stopwords)

palavras_postag_com_stopwords = nltk.pos_tag(palavras_com_stopwords)
apenas_substantivos = [palavra for palavra, tipo in list(filter(lambda tupla_pos: tupla_pos[1] == 'NN', palavras_postag_sem_stopwords))]

freq = FreqDist(apenas_substantivos)

freq.plot(20, cumulative=False)
import pt_core_news_sm

nlp = pt_core_news_sm.load()

palavras_lemmatizadas = []

for p in publicacoes:

    doc = nlp(p)

    [palavras_lemmatizadas.append(token.lemma_) for token in doc]
words = ' '.join(palavras_lemmatizadas)



wordcloud = WordCloud(max_font_size=100,width = 1600, height = 600).generate(words)

plt.figure(figsize=(16,9))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
stemmer = nltk.stem.RSLPStemmer()

palavras_stemmizadas = []

for pub in publicacoes:

    for token in word_tokenize(pub):

        palavras_stemmizadas.append(stemmer.stem(token))
words = ' '.join(palavras_stemmizadas)



wordcloud = WordCloud(max_font_size=100,width = 1600, height = 600).generate(words)

plt.figure(figsize=(16,9))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
nlp = pt_core_news_sm.load()

entidades_reconhecidas = []

for p in publicacoes:

    doc = nlp(p)

    for ent in doc.ents:

        entidades_reconhecidas.append({'entidade': ent.text, 

                                       'posicao': slice(ent.start_char, ent.end_char),

                                       'publicacao': p,

                                      'tipo': ent.label_})

nomes_entidades = {ent['entidade'] for ent in entidades_reconhecidas}

tipos_entidades = {ent['tipo'] for ent in entidades_reconhecidas}

words = ' '.join(nomes_entidades)



wordcloud = WordCloud(max_font_size=100,width = 1600, height = 600).generate(words)

plt.figure(figsize=(16,9))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
words = ' '.join(tipos_entidades)



wordcloud = WordCloud(max_font_size=100,width = 1600, height = 600).generate(words)

plt.figure(figsize=(16,9))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
from gensim import corpora, models


publicacoes_tratadas = []

for f in files[25:100]:

    json_diario = json.load(open(f, 'r', encoding='utf-8'))

    pubs = pd.io.json.json_normalize(json_diario, ['secoes', ['publicacoes']]).values.tolist()

    pub_aux = []

    for p in pubs:

        sem_stopwords = remover_stopwords(word_tokenize(p[0]))

        lemmas = [token.lemma_ for token in nlp(' '.join(sem_stopwords))]

        doc = [stemmer.stem(l) for l in lemmas]

        pub_aux.append(doc)

    

    publicacoes_tratadas.append(pub_aux)
# Estrutura com todas as publicações tratadas de cada diário em uma sublista da lista de diários

publicacoes_tratadas_2_niveis = list(map(lambda x: [' '.join(y) for y in x], publicacoes_tratadas))

publicacoes_tratadas_2_niveis[0]
dictionary = corpora.Dictionary(publicacoes_tratadas_2_niveis)

corpus = [dictionary.doc2bow(text) for text in publicacoes_tratadas_2_niveis]

len(dictionary)
tfidf = models.TfidfModel(corpus)

for document in tfidf[corpus][:10]:

    print({x[1] for x in document})
publicacoes_lda = list(map(lambda x: x.lower(), publicacoes))

publicacoes_lda = [word_tokenize(p) for p in publicacoes_lda]

publicacoes_lda = [remover_stopwords(p) for p in publicacoes_lda]

publicacoes_lda = [','.join(p) for p in publicacoes_lda]
len(publicacoes_lda)
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.feature_extraction.text import CountVectorizer



count_vec = CountVectorizer()

count_lda = count_vec.fit_transform(publicacoes_lda)



n_topicos = 7



lda = LatentDirichletAllocation(n_components=n_topicos)

lda.fit(count_lda)

palavras = count_vec.get_feature_names()

n_palavras = 10

for topic_idx, topic in enumerate(lda.components_):

    print(f"\nTópico {topic_idx+1}\npalavras:")

    palavras_topico = [palavras[i] for i in topic.argsort()[:-n_palavras:-1]]

    print(" ".join([f"{p}," for p in palavras_topico])[:-1])
lda.transform(count_lda)
!pip install PyICU

!pip install pycld2

!pip install morfessor

!polyglot download sentiment2.pt
import polyglot

from polyglot.text import Text, Word
publicacoes_sent = []

for i,p in enumerate(publicacoes):

    t = Text(p)

    sent = sum([w.polarity for w in t.words])

    publicacoes_sent.append((sent, p))
print([s for s, _ in publicacoes_sent])