import warnings, re, string

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

 

from nltk.tokenize import word_tokenize 

from nltk.stem import SnowballStemmer

from nltk.corpus import stopwords



from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression, SGDClassifier

from sklearn.metrics import precision_score, r2_score, recall_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import cross_val_score, KFold

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline 

from sklearn.svm import SVC



from wordcloud import WordCloud 
warnings.filterwarnings('ignore')

pd.set_option('max_colwidth', -1)



Colors = {'Reds':'#e60000', 'Greens':'#4ecc4e'}

Etq = ['Not Disaster', 'Disaster']



PATH_DATA = '../input/nlp-getting-started/'
df = (pd.read_csv(f'{PATH_DATA}train.csv'

                  , sep = ','

                  , header = 0)

      .sample(frac = 1, random_state = 0))

df.head()
def plot(df, col, Var = 'id'):

    df_temp = df.groupby([col], as_index = True).count()[Var].sort_values(ascending = True)

    plt.figure(figsize =(5, 3)) 

    Grafico = df_temp.plot(kind = 'bar'

                       , width = 0.5

                       , color = Colors.values()

                       , stacked = True

                       , legend = False

                       , fontsize = 10)

    

    Grafico.set_ylabel('')

    Grafico.set_xlabel('')

    

    Grafico.grid(axis='y',alpha=0.25)

    Grafico.set_xticklabels(Etq)

    [spine.set_visible(False) for spine in Grafico.spines.values()]

    Grafico.spines['bottom'].set_visible(True)

    

    plt.tick_params(left = False, bottom = False)

    plt.xticks(rotation = 0)

    plt.title(col)

    plt.show()

    plt.close()

    df_temp = None

    

plot(df,'target')
def plotHist(ax, df, filtro, Color, k):

    Grafico(df[df['target']== filtro].text.str.len(),ax, filtro, Color, k)

    

def plotWords(ax, df, filtro, Color, k):

    Grafico(df[df['target']== filtro].text.str.split().map(lambda x: len(x)),ax, filtro, Color, k)

    

def Grafico(serie, ax, filtro, Color, k):

    ax.hist(serie, color = Color)

    [spine.set_visible(False) for spine in ax.spines.values()]

    ax.tick_params(left = False, bottom = False)

    ax.spines['bottom'].set_visible(True)

    ax.set_title(k)

    ax.grid(axis='y',alpha=0.25)
fig, ax = plt.subplots(1, 4, figsize = (16,3))

fig.suptitle('Caracteres & palabras x tweet', x = 0.5, y = 1.1)

fig.patch.set_facecolor('white')



for Color, i , k in zip(Colors.values(), range(2), Etq):

    plotHist(ax[i], df, (i + 1) % 2, Color, k)

    

for Color, i, k in zip(Colors.values(), range(2, 4), Etq):

    plotWords(ax[i], df, (i + 1) % 2, Color, k)



plt.show()
def clean_df(df):

    df.text = df.text.apply(lambda x: x.lower())

    df.text = df.text.apply(lambda x: re.sub(r'https?://\S+|www\.\S+', '', x))

    df.text = df.text.apply(lambda x: re.sub(r'\[.*?\]', '', x))

    df.text = df.text.apply(lambda x: re.sub(r'<.*?>', '', x))

    df.text = df.text.apply(lambda x: re.sub(r'[%s]' % re.escape(string.punctuation), '', x))

    df.text = df.text.apply(lambda x: re.sub(r'\n', '', x))

    df.text = df.text.apply(lambda x: re.sub(r'\w*\d\w*', '', x))

    

    return df



df = clean_df(df)



df.head()
Punct_List = dict((ord(punct), None) for punct in string.punctuation )

Stemmer_List = SnowballStemmer('english')



def TxNormalize(text):

    tokens = word_tokenize(str(text).replace('/',' ').translate(Punct_List))

    return [x for x in tokens if x not in stopwords.words('english')]



def Crear_WordCloud(ax, tokens, Color, Titulo, Theme):

    if len(tokens) > 0:

        

        wc = WordCloud(width = 6000

                       , height = 3500

                       , min_font_size = 60

                       , max_words = 100

                       , background_color = 'white'

                       , colormap = Theme

                       , random_state = 0

                      ).generate(tokens) 

        

        ax.imshow(wc)

        ax.set_title(Titulo, fontsize = 60, color = Color, y = 1.03)

        ax.axis('off')

        

    

def Plot(ax, df, Color, Titulo):

    datos = df.word.value_counts(sort = True).nlargest(25)

    ax.barh(datos.index, datos.values, color = Color)

    

    ax.tick_params(left = False, bottom = False)

    ax.invert_yaxis()

    [spine.set_visible(False) for spine in ax.spines.values()]

    ax.spines['left'].set_visible(True)

    ax.set_title(Titulo.capitalize(), fontsize = 14)

    ax.grid(axis = 'x', alpha = 0.25)

        

def tokenize(df, filtro):

    tokens = []



    for i in df[df.target == filtro].text:

        tokens += TxNormalize(i)

        

    return tokens
Words = [

    (df[(df.target == i)].text

     .apply(TxNormalize)

     .explode()

     .str.cat(sep = ' '))

    for i in range(2)]
fig, axs = plt.subplots(nrows = 2, ncols = 1, figsize = (20, 22))

fig.patch.set_facecolor('white')



for ax, Color, Token, Theme, i in zip(axs, Colors.values(), Words, Colors.keys(), Etq):

    Crear_WordCloud(ax, Token, Color, i, Theme)

    

plt.show()

plt.close()
fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 12))

fig.patch.set_facecolor('white')



for ax, i, Color, tokens in zip(axs, Etq, Colors.values(), Words):

    Plot(ax, pd.DataFrame(tokens.split(' '), columns = ['word']), Color, i)

    

fig.suptitle('Most frecuent words', fontsize = 20)

plt.show()

plt.close()
Params = {'Regresión Logística': LogisticRegression(C = 1e5

                                                    , class_weight = 'balanced'

                                                    , random_state = 0)

          

           , 'Pasive Aggressive Classifier': PassiveAggressiveClassifier(max_iter = 50

                                                                         , random_state = 0)

          

           , 'SGD': SGDClassifier(max_iter = 5

                                  , tol = None

                                  , random_state = 0 )

          

           , 'Naive Bayes': MultinomialNB(alpha = 1)

          

           , 'Support Vector Machine': SVC(kernel = 'linear'

                                           , C = 1

                                           , gamma = 'scale' 

                                           , random_state = 0 )}



def Modelo(Clf, X, y):

    clf = Pipeline([

        ('tfidf', TfidfVectorizer(ngram_range = (1, 4)

                                  , analyzer = 'word'

                                  , stop_words = 'english'))        

        , ('Classifier', Clf)

    ])

    

    return clf.fit(X, y)
X = df.text

y = df.target



Result = {}

kf = KFold(n_splits = 10)

                      

for i in Params:

    Score = []

    clf = Modelo(Params[i], X, y)

    

    Score.append(Params[i].__class__.__name__)

    Score.append(cross_val_score(clf, X, y, cv = kf, scoring = 'accuracy').mean())

    Result[i] = Score 





Models = (pd.DataFrame(Result, index=['Tipo', 'Cross Val'])

          .transpose()

          .sort_values(by = 'Cross Val', ascending = False)

          .reset_index()

          .rename(columns = {'index':'Modelo'}))



Models
df = (pd.read_csv(f'{PATH_DATA}test.csv'

                  , sep = ','

                  , header = 0))



df = clean_df(df)

df.head()
Best = Modelo(Params[Models['Modelo'][0]], X, y)  

df['target'] = Best.predict(df.text)
df[['id','target']].to_csv('submission.csv', index = False)