!pip install feedparser

import feedparser

import time

import pandas as pd

import matplotlib.pyplot as plt

# sklearn

# ML classificators

from sklearn.linear_model import SGDClassifier

# ML selecao de dados de treino e teste

from sklearn.model_selection import train_test_split, cross_val_score

# confusion matrics

from sklearn.metrics import confusion_matrix

# metrics

from sklearn import metrics

# vetorizador

from sklearn.feature_extraction.text import TfidfVectorizer
# get news

rssAll = {

    'nfl': 'https://www.espn.com/espn/rss/nfl/news',

    'nba': 'https://www.espn.com/espn/rss/nba/news',

    'motor': 'https://www.espn.com/espn/rss/rpm/news',

    'futebol': 'https://www.espn.com/espn/rss/soccer/news',

    'mlb': 'https://www.espn.com/espn/rss/mlb/news',

    'nhl': 'https://www.espn.com/espn/rss/nhl/news',

    'Poker': 'https://www.espn.com/espn/rss/poker/master'

}



# creating dataframe

tipo = []

titulo = []

texto = []

for i in rssAll.items():

    for j in feedparser.parse(i[1]).entries:

        tipo.append(i[0])

        titulo.append(j['title'])

        texto.append(j['title'] + '. ' + j['summary'])

df = pd.DataFrame({'tipo': tipo, 'titulo': titulo, 'miniNews': texto})

# get uniques news categories

tipos = df['tipo'].unique()
# showing categories news (uniques)

tipos
# creating words dictionary

textos = df['miniNews']

palavras = textos.str.lower().str.split()



dicionario = set()

lista = []

for i in palavras:

    dicionario.update(i)

    for j in i:

        lista.append(j)



palavraEposicao = dict(zip(dicionario, range(len(dicionario))))
palavraEposicao
def textToNro(txt):

    global tipos

    return list(tipos).index(txt)





df['tipoN'] = df['tipo'].apply(textToNro)
# Split data train and test using sklearn

X = textos

y = df.tipoN



Xtreino, Xteste, ytreino, yteste = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
# vectorizing data train

txtvetorizado = TfidfVectorizer()

vetorXtreino = txtvetorizado.fit_transform(Xtreino)
# training data

#

modelo = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)

modelo.fit(vetorXtreino, ytreino)
# vectorizing data test

vetorXteste = txtvetorizado.transform(Xteste)
# predicting

previsao = modelo.predict(vetorXteste)
# showing metrics

metrics.classification_report(yteste.values, previsao, target_names=tipos)
# analysing predicts data with confusion matrix

confusion_matrix = confusion_matrix(yteste.values, previsao)

pd.crosstab(yteste.values, previsao, rownames=['Real'], colnames=['Previsto'], margins=True)
texto = []

# reading rss

rssHighLights = {'ESPN': 'https://www.espn.com/espn/rss/news'}

dfespn = pd.DataFrame(columns=['mininews'])

for i in rssHighLights.items():

    for j in feedparser.parse(i[1]).entries:

        texto.append(j['title'] + '. ' + j['summary'])



# saving rss news in dataframe

dfespn['mininews'] = texto
# showing 5 examples

dfespn.sample(5)
# vectorize the news read.

novoVetor = txtvetorizado.transform(dfespn['mininews'])
# predicting the news

previsoes = modelo.predict(novoVetor)
for noticia, tipoNoticia in zip(dfespn['mininews'], previsoes):

    print(f'{noticia}: ')

    print(f'PREDICT==> {tipos[tipoNoticia].upper()}')

    time.sleep(1)
import textblob

from textblob import TextBlob



def getSentiment(txt=''):

    txt = TextBlob(txt)

    return txt.sentiment.polarity
cont = []

for i in tipos:

    cont.append(len(df[df['tipo'] == i]))



fig1, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 8))



ax1.pie(cont, labels=tipos, autopct='%1.1f%%')

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

ax1.set_title('FACT: Categorias nos feeds (%)')



p = n = t = 0



for i in range(0, len(df)):

    polarity = getSentiment(df.iloc[i]['miniNews'])

    if polarity != 0:

        t += 1

        if polarity > 0:

            p += 1

        else:

            n += 1



ax2.pie([p, n], labels=['positive', 'negative'], autopct='%1.1f%%')

ax2.axis('equal')  

ax2.set_title('FACT: Humor (%)')



cont = []

dfespn['tipo'] = previsoes

for c, v in enumerate(tipos):

    cont.append(len(dfespn[dfespn['tipo'] == c]))

ax3.pie(cont, labels=tipos, autopct='%1.1f%%')

ax3.axis('equal')  

ax3.set_title('PREDICT: LAST NEWS CATEGORIES ')



p = n = t = 0



for i in range(0, len(dfespn)):

    polarity = getSentiment(dfespn.iloc[i]['mininews'])

    if polarity != 0:

        t += 1

        if polarity > 0:

            p += 1

        else:

            n += 1



ax4.pie([p, n], labels=['positive', 'negative'], autopct='%1.1f%%')

ax4.axis('equal')  

ax4.set_title('PREDICT: LAST NEWS HUMOR RATE (%)')