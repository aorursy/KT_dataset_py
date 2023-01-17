import pandas as pd

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.probability import FreqDist

from textblob import TextBlob
import math
import pickle
import string
import seaborn as sns
import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
# from sklearn.metrics import f1_score, accuracy_score
# from sklearn.model_selection import cross_val_score
# from sklearn.svm import LinearSVC
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import Pipeline
#data = pd.read_csv('./titles_sentiment.csv')
data = pd.read_csv('../input/titles_tagged_metaphors.csv')
stdout = pd.read_csv('../input/sentences_metaphors_stdout.csv')
try:
    pass#data = data.set_index('id')
except:
    pass
sw = list(set(stopwords.words('english')))
types = list(set(data.type.values))
qtd_type = [ len(data[data['type'] == t]) for t in types ]
ax = sns.barplot(x=types, y=qtd_type)
for item in ax.get_xticklabels():
    item.set_rotation(45)
stemmer = PorterStemmer()
tokens = dict()
for t in types:
    if not t in tokens:
        tokens[t] = []
    titles = data[data['type'] == t].title.values
    
    for title in titles:
        tokenized = word_tokenize(title)
        tokenized = [stemmer.stem(item.lower()) for item in tokenized if item.lower() not in sw]
        tokens[t] += tokenized
freqs = dict()
for t in types:
    freqs[t] = FreqDist(tokens[t])
def dict_filter (word_freq):
    return dict( (word,word_freq[word]) for word in word_freq if word.isalnum() )

for f in freqs:
    freqs[f] = FreqDist( dict_filter(freqs[f]) )
    freqs[f].plot(20, title=f)
#Contagem de tokens
all_tokens = []
for t in tokens:
    all_tokens += tokens[t]
#Proporção de metáforas
token_count = len(all_tokens)
metaphor_count = sum(data['detected_metaphors'].values)
print("Proporção de tokens com metáforas em relação à quantidade total de tokens: %.2f%%" % ((metaphor_count/token_count)*100))
#para cada classe
#textblob polaridade absoluta

#avg_abs_textblob_polarity: é a média de polaridade para a classe
#avg_textblov_subjectivity: é a média de subjetividade para a classe
#avg_vader_neg: é a média de sentimento negativo para a classe
#avg_vader_pos: é a média de sentimento positivo para a classe
#strong_polarity_percent: é a porcentagem de registros que contem polaridade forte ( > 0.5) para a classe
#strong_subjectivity_percent: é a porcentagem de registros que contem subjetividade forte (> 0.5) para a classe
#strong_neg_percent: é a porcentagem de registros que contem negatividade forte (> 0.5) para a classe
#strong_pos_percent: é a porcentagem de registros que contem positividade forte (> 0.5) para a classe
#agv_metaphors: média da quantidade de metáforas detectadas para a classe
#max_metaphors: valor máximo de metáforas detectadas em um mesmo título, para a classe

#As demais estatísticas que contém _metaphors são semelhantes às descritas acima, mas consideram apenas
#títulos com pelo menos uma metáfora

sent_analysis = {}
for t in types:
    sent_analysis[t] = {
        'avg_abs_textblob_polarity': data[data['type'] == t]['textblob_polarity'].apply(abs).mean(),
        'avg_textblov_subjectivity': data[data['type'] == t]['textblob_subjectivity'].apply(abs).mean(),
        'avg_vader_neg': data[data['type'] == t]['vader_neg'].apply(abs).mean(),
        'avg_vader_pos': data[data['type'] == t]['vader_pos'].apply(abs).mean(),
        'avg_vader_neu': data[data['type'] == t]['vader_neu'].apply(abs).mean(),
        'strong_polarity_percent': len(data[(data['type'] == t) & (data['textblob_polarity'].apply(abs) > .5)]['textblob_polarity'])/len(data[data['type'] == t]),
        'strong_subjectivity_percent': len(data[(data['type'] == t) & (data['textblob_subjectivity'].apply(abs) > .5)]['textblob_subjectivity'])/len(data[data['type'] == t]),
        'strong_neg_percent': len(data[(data['type'] == t) & (data['vader_neg'].apply(abs) > .5)]['vader_neg'])/len(data[data['type'] == t]),
        'strong_pos_percent': len(data[(data['type'] == t) & (data['vader_pos'].apply(abs) > .5)]['vader_pos'])/len(data[data['type'] == t]),
                        
        'avg_metaphors': data[data['type'] == t]['detected_metaphors'].mean(),
        'max_metaphors': data[data['type'] == t]['detected_metaphors'].max(),
                        
        'avg_abs_textblob_polarity_metaphors': data[(data['type'] == t) & (data['detected_metaphors'] > 0)]['textblob_polarity'].apply(abs).mean(),
        'avg_textblov_subjectivity_metaphors': data[(data['type'] == t) & (data['detected_metaphors'] > 0)]['textblob_subjectivity'].apply(abs).mean(),
        'avg_vader_neg_metaphors': data[(data['type'] == t) & (data['detected_metaphors'] > 0)]['vader_neg'].apply(abs).mean(),
        'avg_vader_pos_metaphors': data[(data['type'] == t) & (data['detected_metaphors'] > 0)]['vader_pos'].apply(abs).mean(),
        'strong_polarity_percent_metaphors': len(data[(data['type'] == t) & (data['detected_metaphors'] > 0) & (data['textblob_polarity'] > .5)])/len(data[(data['type']==t) & (data['detected_metaphors'] > 0)]),
        'strong_subjectivity_percent_metaphors': len(data[(data['type'] == t) & (data['detected_metaphors'] > 0) & (data['textblob_subjectivity'] > .5)])/len(data[(data['type']==t) & (data['detected_metaphors'] > 0)]),
        'strong_neg_percent_metaphors': len(data[(data['type'] == t) & (data['detected_metaphors'] > 0) & (data['vader_neg'] > .5)])/len(data[(data['type']==t) & (data['detected_metaphors'] > 0)]),
        'strong_pos_percent_metaphors': len(data[(data['type'] == t) & (data['detected_metaphors'] > 0) & (data['vader_pos'] > .5)])/len(data[(data['type']==t) & (data['detected_metaphors'] > 0)])
    }
def barplot_statistics(prop, plot_title):
    statistic = [sent_analysis[t][prop] for t in types ]
    for t in types:
        ax = sns.barplot(x=types, y=statistic)
        ax.set_title(plot_title)
        for item in ax.get_xticklabels():
            item.set_rotation(45)
#Média de polaridade absoluta
barplot_statistics('avg_abs_textblob_polarity', 'Média de polaridade absoluta')
#Média absoluta de subjetividade
barplot_statistics('avg_textblov_subjectivity', 'Média de subjetividade')
#Média absoluta de negatividade
barplot_statistics('avg_vader_neg', 'Média de negatividade')
#Média absoluta de positividade
barplot_statistics('avg_vader_pos', 'Média de positividade')
#Média absoluta de neutralidade
barplot_statistics('avg_vader_neu', 'Média de neutralidade')
#Média absoluta de polaridade forte
barplot_statistics('strong_polarity_percent', 'Porcentagem de títulos com polaridade forte')
#Média absoluta de subjetividade forte
barplot_statistics('strong_subjectivity_percent', 'Porcentagem de títulos com subjetividade forte')
#Média absoluta de positividade forte
barplot_statistics('strong_pos_percent', 'Porcentagem de títulos com positividade forte')
#Média de metáforas por classe
barplot_statistics('avg_metaphors', 'Média de metáforas por classe')
#Quantidade Máxima de metáforas em um título, por classe
barplot_statistics('max_metaphors', 'Quantidade Máxima de metáforas em um título')
#Média de polaridade absoluta para títulos com metáforas
barplot_statistics('avg_abs_textblob_polarity_metaphors', 'Média de polaridade absoluta para títulos com metáforas')
#Média absoluta de subjetividade para títulos com metáforas
barplot_statistics('avg_textblov_subjectivity_metaphors', 'Média de subjetividade para títulos com metáforas')
#Média absoluta de negatividade para títulos com metáforas
barplot_statistics('avg_vader_neg_metaphors', 'Média de negatividade para títulos com metáforas')
#Média absoluta de positividade para títulos com metáforas
barplot_statistics('avg_vader_pos_metaphors', 'Média de positividade para títulos com metáforas')
#Média absoluta de polaridade forte para títulos com metáforas
barplot_statistics('strong_polarity_percent_metaphors', 'Porcentagem de títulos com polaridade forte para títulos com metáforas')
#Média absoluta de subjetividade forte para títulos com metáforas
barplot_statistics('strong_subjectivity_percent_metaphors', 'Porcentagem de títulos com subjetividade forte para títulos com metáforas')
#Média absoluta de negatividade forte para títulos com metáforas
barplot_statistics('strong_neg_percent_metaphors', 'Porcentagem de títulos com Negatividade forte para títulos com metáforas')
#Média absoluta de positividade forte para títulos com metáforas
barplot_statistics('strong_pos_percent_metaphors', 'Porcentagem de títulos com positividade forte para títulos com metáforas')
#filtra palavras metafóricas
met_stdout = stdout#[stdout['has_metaphor'] == 1]
#transforma o dataset para om dict, para que as metáforas sejam marcadas
titles_dict = data.set_index('id').to_dict('index')
#para cada título, encontra a metáfora e marca com um M_
for item_metaphor in met_stdout.values:
    try:
        if (item_metaphor[1]==1):
            titles_dict[ item_metaphor[0] ]['title'] = titles_dict[ item_metaphor[0] ]['title'].replace(item_metaphor[2], "M_"+str(item_metaphor[2]))
        titles_dict[ item_metaphor[0] ]['id'] = item_metaphor[0]
    except:
        pass#print(item_metaphor[0])
#transforma o dict para um df
data_with_metaphors = pd.DataFrame(list(titles_dict.values()))
data_with_metaphors.set_index('id').to_csv('titles_tagged_metaphors.csv')
#taggeddata = pd.read_csv('../input/titles_tagged_metaphors.csv',usecols=['id','title','type','detected_metaphors'])
taggeddata = pd.read_csv('../input/titles_tagged_metaphors.csv')
taggeddata.head()
#recupera os títulos originais
#taggeddata['title_orig'] = taggeddata.apply(lambda row: row['title'].replace("M_",""), axis=1)
taggeddata.head()
#o POS deve ser feito em cima de todo o título, sem a tag M_
def word_man(sentence):
    tokenSemTagM = nltk.word_tokenize(sentence.replace("M_",""))
    tokenComTagM = nltk.word_tokenize(sentence)
    #print(len(tokenComTagM))
    pos = nltk.pos_tag(tokenSemTagM)
    #print(len(pos))
    if len(pos) != len(tokenSemTagM) != len(tokenComTagM):
        print('ops')
    result = []
    try:
        for i in range(len(tokenComTagM)):
            if "M_" in tokenComTagM[i]:
                result.append(pos[i])
    except:
        return []
    return result

taggeddata['met_tokenized_sents'] = taggeddata.apply(lambda row: word_man(row['title']), axis=1)

taggeddata.head()
# inicialmente vamos abstrair várias categorias de NN em uma única e várias categorias de VB em uma única
typesLang = ['NN', 'VB']

for t in typesLang:
    def word_manNN(tokenized_sents):
        try:
            for i in range(len(tokenized_sents)):
                if t in tokenized_sents[i][1]:
                    return 1
        except:
            return 0
        return 0

    taggeddata['met_'+t] = taggeddata.apply(lambda row: word_manNN(row['met_tokenized_sents']), axis=1)
taggeddata.head()
taggeddata.set_index('id').to_csv('titles_tagged_metaphors_with_tokenized_sents.csv')
#len(taggeddata[(taggeddata['type'] == 'fake') & (taggeddata['met_NN'] == 0) & (taggeddata['met_VB'] == 0)])
#len(taggeddata[(taggeddata['type'] == 'fake') & ((taggeddata['met_NN'] == 1) | (taggeddata['met_VB'] == 1))])
#len(taggeddata[(taggeddata['type'] == 'fake') & (taggeddata['met_NN'] == 1) & (taggeddata['met_VB'] == 0)])
#len(taggeddata[(taggeddata['type'] == 'fake') & (taggeddata['met_NN'] == 0) & (taggeddata['met_VB'] == 1)])
#len(taggeddata[(taggeddata['type'] == 'fake') & (taggeddata['met_NN'] == 1) & (taggeddata['met_VB'] == 1)])

types
sent_analysis = {}
for t in types:
    lenTotal = len(taggeddata[(taggeddata['type'] == t) & (taggeddata['detected_metaphors'] == 1)])
    print(lenTotal)
    sent_analysis[t] = {
        'abs_sem_metafora' : len(taggeddata[(taggeddata['type'] == t) & (taggeddata['met_NN'] == 0) & (taggeddata['met_VB'] == 0)]),
        'abs_com_metafora' : len(taggeddata[(taggeddata['type'] == t) & ((taggeddata['met_NN'] == 1) | (taggeddata['met_VB'] == 1))]),
        'abs_com_metafora_so_NN' : len(taggeddata[(taggeddata['type'] == t) & (taggeddata['met_NN'] == 1) & (taggeddata['met_VB'] == 0)]),
        'abs_com_metafora_so_VB' : len(taggeddata[(taggeddata['type'] == t) & (taggeddata['met_NN'] == 0) & (taggeddata['met_VB'] == 1)]),
        'abs_com_metafora_NN_e_VB' : len(taggeddata[(taggeddata['type'] == t) & (taggeddata['met_NN'] == 1) & (taggeddata['met_VB'] == 1)]),
        
        'pct_sem_metafora' : (len(taggeddata[(taggeddata['type'] == t) & (taggeddata['detected_metaphors'] == 1) & (taggeddata['met_NN'] == 0) & (taggeddata['met_VB'] == 0)])/lenTotal),
        'pct_com_metafora' : (len(taggeddata[(taggeddata['type'] == t) & (taggeddata['detected_metaphors'] == 1) & ((taggeddata['met_NN'] == 1) | (taggeddata['met_VB'] == 1))])/lenTotal),
        'pct_com_metafora_so_NN' : (len(taggeddata[(taggeddata['type'] == t) & (taggeddata['detected_metaphors'] == 1) & (taggeddata['met_NN'] == 1) & (taggeddata['met_VB'] == 0)])/lenTotal),
        'pct_com_metafora_so_VB' : (len(taggeddata[(taggeddata['type'] == t) & (taggeddata['detected_metaphors'] == 1) & (taggeddata['met_NN'] == 0) & (taggeddata['met_VB'] == 1)])/lenTotal),
        'pct_com_metafora_NN_e_VB' : (len(taggeddata[(taggeddata['type'] == t) & (taggeddata['detected_metaphors'] == 1) & (taggeddata['met_NN'] == 1) & (taggeddata['met_VB'] == 1)])/lenTotal)
    }
barplot_statistics('abs_sem_metafora', 'Total sem metáfora') # TODO AJUSTAR

barplot_statistics('abs_com_metafora', 'Total com metáfora (NN ou VB)') #AJUSTAR
barplot_statistics('abs_com_metafora_so_NN', 'Total com metáfora (somente NN)')
barplot_statistics('abs_com_metafora_so_VB', 'Total com metáfora (somente VB)')

barplot_statistics('abs_com_metafora_NN_e_VB', 'Total com metáfora (NN e VB presentes)')
barplot_statistics('pct_sem_metafora', 'Percentual de metáforas que não são um VB ou NN') # AJUSTAR
barplot_statistics('pct_com_metafora', 'Percentual de NN ou VB nos títulos com metáfora ') #AJUSTAR

barplot_statistics('pct_com_metafora_so_NN', 'Percentual de NN nos títulos com metáfora')

barplot_statistics('pct_com_metafora_so_VB', 'Percentual de VB nos títulos com metáfora')
barplot_statistics('pct_com_metafora_NN_e_VB', 'Percentual de NN e VB em conjunto, nos títulos com metáfora')







