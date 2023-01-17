import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings, re, string



from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize 



from wordcloud import WordCloud



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline
warnings.filterwarnings('ignore')

pd.set_option('max_colwidth', -1)



Sents = ['negative','neutral','positive']

Colors = {'Reds':'#e60000'

          , 'Greys':'#a6a6a6'

          , 'Greens':'#4ecc4e'}



PATH_DATA = '../input/tweet-sentiment-extraction/'
df = (pd.read_csv(f'{PATH_DATA}train.csv'              

                  , sep = ','

                  , header = 0)

      .fillna(''))



test = (pd.read_csv(f'{PATH_DATA}test.csv')

        .fillna(''))



df.head()
Grafico = (df.groupby(['sentiment'], as_index = True)

           .count()['textID']).plot(kind = 'bar'

                                    , width = 0.5

                                    , color = Colors.values()

                                    , stacked = True

                                    , legend = False

                                    , fontsize = 10

                                    , figsize = (5, 3))



[spine.set_visible(False) for spine in Grafico.spines.values()]



Grafico.spines['bottom'].set_visible(True)

Grafico.grid(axis = 'y', alpha = 0.25)

Grafico.set_ylabel('')

Grafico.set_xlabel('')



plt.tick_params(left = False, bottom = False)

plt.xticks(rotation = 0)

plt.title('sentiment')

plt.show()

plt.close()    
def plotHist(ax, df, Sent, Color):

    Grafico(df[df.sentiment == Sent].text.str.len(),ax, Color)

    

def plotWords(ax, df, Sent, Color):

    Grafico(df[df.sentiment == Sent].text.apply(lambda x: len(str(x).split())),ax, Color)

    

def Grafico(serie, ax, Color):

    ax.hist(serie, color = Color)

    [spine.set_visible(False) for spine in ax.spines.values()]

    ax.tick_params(left = False, bottom = False)

    ax.spines['bottom'].set_visible(True)

    ax.grid(axis = 'y', alpha = 0.25)

    

fig, axs = plt.subplots(1, 3, figsize = (12,3))

for ax, Sent, Color in zip(axs, Sents, Colors.values()):

    plotHist(ax, df, Sent, Color)



fig.suptitle('Caracteres x tweet', x = 0.07, y=0.72, rotation = 90)

plt.show()

plt.close()



fig, axs = plt.subplots(1, 3, figsize = (12,3))

for ax, Sent, Color in zip(axs, Sents, Colors.values()):

    plotWords(ax, df, Sent, Color)



fig.suptitle('Words x tweet'

             , x = 0.07

             , y = 0.72

             , rotation = 90)

plt.show()

plt.close()
Punct_List = dict((ord(punct), None) for punct in string.punctuation)



def TxNormalize(text):

    text = text.lower()

    tokens = word_tokenize(str(text).replace('/',' ').translate(Punct_List))

    return [x for x in tokens if x not in stopwords.words('english') + ['u', 'im']]



def tokenize(df, filtro):

    tokens = []

    for i in df[(df.sentiment == filtro) & (df.Val == 0)].text:

        tokens += TxNormalize(i)

    return tokens



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

        ax.set_title(Titulo, fontsize = 60, color = Color)

        ax.axis('off')



def Plot(ax, df, Color, Sent):

    datos = df.word.value_counts(sort = True).nlargest(25)

    ax.barh(datos.index, datos.values, color = Color)

    

    ax.tick_params(left = False, bottom = False)

    ax.invert_yaxis()

    [spine.set_visible(False) for spine in ax.spines.values()]

    ax.spines['left'].set_visible(True)

    ax.set_title(Sent.capitalize(), fontsize = 14)

    ax.grid(axis = 'x', alpha = 0.25)
Words = [

    (df[(df.sentiment == Sent)].text

     .apply(TxNormalize)

     .explode()

     .str.cat(sep = ' '))

    for Sent in Sents]



fig, axs = plt.subplots(nrows = 3, ncols = 1, figsize = (20, 32))



for ax, Sent, Color, Token, Theme in zip(axs, Sents, Colors.values(), Words, Colors.keys()):

    Crear_WordCloud(ax, Token, Color, Sent.capitalize(), Theme)

    

plt.show()

plt.close()
fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (18, 12))



for ax, Sent, Color, tokens in zip(axs, Sents, Colors.values(), Words):

    Plot(ax, pd.DataFrame(tokens.split(' '), columns = ['word']), Color, Sent)

    

fig.suptitle('Most frecuent words', fontsize = 20)

plt.show()

plt.close()
def createModel():

    Pipe = Pipeline(steps = [('Vec', CountVectorizer(tokenizer = lambda x: x.split()))

                             , ('Clf', SGDClassifier(max_iter = 1000, random_state = 0))])

    

    return Pipe



X = pd.concat([df.text, test.text], axis=0)

y = pd.concat([df.sentiment, test.sentiment], axis=0)



Model = createModel().fit(X, y)

pred = Model.predict(X)



print(classification_report(y_true = y, y_pred = pred))

print('\nShape:', X.shape, y.shape)
class WordSelector(BaseEstimator, TransformerMixin):

    def __init__(self, pos_class_std = 2.2, neg_class_std = 2.2):

        self.scores = {}

        self.pos_class_std = pos_class_std

        self.neg_class_std = neg_class_std



        self.vocabulary_ = Model.named_steps['Vec'].vocabulary_

        self.coef_ = Model.named_steps['Clf'].coef_ 

        

        self.weights_by_classes = {

            'negative': list(enumerate(self.coef_[0]))

            , 'neutral':  list(enumerate(self.coef_[1]))

            , 'positive': list(enumerate(self.coef_[2]))

        }

  

    def get_weights(self, text_list, class_weights):

        text_idx = [self.vocabulary_[tok.lower()] for tok in text_list if tok.lower() in self.vocabulary_]

        

        return [class_weights[idx][1] for idx in text_idx]



    def get_top_words(self, words_list, weights_list, num_std):

        mean, std, top_words = [np.mean(weights_list), np.std(weights_list), []]

        

        for word, weight in zip(words_list, weights_list):

            if weight > (mean +  num_std * std):

                top_words.append(word)

                

        return ' '.join(top_words)





    def select_words(self, df):  

        text, sentiment = df



        if sentiment == 'neutral':

            return text

        elif sentiment == 'positive':

            num_std = self.pos_class_std

        else:

            num_std = self.neg_class_std   

            

        text = ' '.join(re.sub('(\w+:\/\/\S+)', ' ', text).split()).split()

        weights = self.get_weights(text, self.weights_by_classes[sentiment])

        res = self.get_top_words(text, weights, num_std = num_std)



        return ' '.join(text) if res == '' else res

    

    def fit(self, X, y = None):

        return self

    

    def predict(self, X):



        df = X

        df['selected_text'] = X[['text', 'sentiment']].apply(self.select_words, axis=1)



        return df.selected_text

    

    def jaccard(self, df):

        

        a = set(df.predictions.lower().split()) 

        b = set(df.selected_text.lower().split())

        

        if len(a) + len(b) == 0:

            return 0.5

        c = a.intersection(b)

        return float(len(c)) / (len(a) + len(b) - len(c))



    def score(self, X, y):

        df = X

        df['selected_text'] = y

        df['predictions'] = self.predict(df[['text', 'sentiment']])

        

        df['score'] = df[['predictions', 'selected_text']].apply(self.jaccard, axis=1)



        return round(df.score.mean(), 4)
parameters = {

    'pos_class_std': [1.90, 1.93, 1.95, 1.98]

    , 'neg_class_std': [2.28, 2.3, 2.32, 2.34]

}



gs = GridSearchCV(WordSelector()

                  , parameters

                  , cv = 5

                  , verbose = 1

                  , n_jobs = -1

                 )



gs = gs.fit(df[['text', 'sentiment']], df['selected_text'])



gs.best_params_
word_selector = WordSelector(pos_class_std = gs.best_params_['pos_class_std']

                             , neg_class_std = gs.best_params_['neg_class_std'])



df['predictions'] = word_selector.predict(df[['text', 'sentiment']])

df['score'] = df[['predictions', 'selected_text']].apply(word_selector.jaccard, axis=1)
def Matriz(df, Score):

    df = (df.groupby('sentiment')['score'].agg([np.sum, np.size])

          .reset_index())

    

    df['1'] = df['size'] - df['sum']

    

    Grafico = (df[['sentiment','sum', '1']]

               .set_index('sentiment')

               .reindex(columns=['sum', '1'])).plot(kind = 'bar'

                                                    , width = 0.5

                                                    , color = [['#00acee','#00acee','#00acee'], Colors.values()]

                                                    , stacked = True

                                                    , legend = False

                                                    , fontsize = 10

                                                    , figsize = (5, 3))



    [spine.set_visible(False) for spine in Grafico.spines.values()]



    Grafico.spines['bottom'].set_visible(True)    

    Grafico.grid(axis = 'y', alpha = 0.25)

    Grafico.set_ylabel('')

    Grafico.set_xlabel('')



    Grafico.legend({'Jaccard: ' + str(round(Score * 100,2)) + ' %'}

                   , loc = 'center'

                   , bbox_to_anchor = (0.5, 1.13)

                   , ncol = 1

                   , frameon = False

                   , fontsize = 12)



    plt.tick_params(left = False, bottom = False)

    plt.xticks(rotation = 0)

    plt.show()

    plt.close()



Matriz(df, word_selector.score(df[['text', 'sentiment']], df.selected_text))
test['selected_text'] = word_selector.predict(test[['text', 'sentiment']])



test[['textID','selected_text']].to_csv('submission.csv', index=False, header=True)



test[['textID','selected_text']].head()