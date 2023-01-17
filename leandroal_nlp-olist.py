import pandas as pd

pd.set_option('max_colwidth', 400)

import numpy as np

import re

from unidecode import unidecode



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, log_loss,confusion_matrix



import spacy

import nltk

from nltk import FreqDist

from gensim.models.phrases import Phrases, Phraser

from gensim.models import Word2Vec



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

#plt.style.use('seaborn')

import seaborn as sns

%matplotlib inline



import  warnings

warnings.filterwarnings("ignore")
# =============================================================================

# Importação de Todos os Sub-datasets

# =============================================================================

files = {'customers'    : '/kaggle/input/brazilian-ecommerce/olist_customers_dataset.csv',

         'geolocation'  : '/kaggle/input/brazilian-ecommerce/olist_geolocation_dataset.csv',

         'items'        : '/kaggle/input/brazilian-ecommerce/olist_order_items_dataset.csv',

         'payment'      : '/kaggle/input/brazilian-ecommerce/olist_order_payments_dataset.csv',

         'orders'       : '/kaggle/input/brazilian-ecommerce/olist_orders_dataset.csv',

         'products'     : '/kaggle/input/brazilian-ecommerce/olist_products_dataset.csv',

         'sellers'      : '/kaggle/input/brazilian-ecommerce/olist_sellers_dataset.csv',

         'review'       : '/kaggle/input/brazilian-ecommerce/olist_order_reviews_dataset.csv',

         }



dfs = {}

for key, value in files.items():

    dfs[key] = pd.read_csv(value)

    

for k, v in dfs.items():

    print(f'{k}: {v.shape}')
# =============================================================================

# Seleção do Dataset relevante

# =============================================================================

df = dfs['customers'].merge(dfs['orders'], how='left', on='customer_id')

df = df.merge(dfs['review'], how='left', on='order_id')



np.random.seed(100)

df.sample(1).T
# =============================================================================

# Comprimento das sentenças

# =============================================================================

df['review_comment_message'].str.len().describe()
print('\nFormato do dataset antes da remoção de duplicados e nulos:', df.shape)



# =============================================================================

# Quantidade de textos vazios

# =============================================================================

print('Avaliações nulas: ', df['review_comment_message'].isnull().sum())



# =============================================================================

# Quantidade de registros duplicados

# =============================================================================

print('Registros duplicados: ', df['review_comment_message'].duplicated(keep=False).sum())



# =============================================================================

# Adequações finais

# =============================================================================

df = df[~df['review_comment_message'].isna()].reset_index(drop=True)

df = df[df['review_comment_message'].str.contains("\w")]

df = df[df['review_comment_message'].str.len() > 5]

df = df.drop_duplicates('review_comment_message').reset_index(drop=True)



print('\nFormato do dataset após remoção de duplicados e nulos:', df.shape)

print('Avaliações nulas: ', df['review_comment_message'].isnull().sum())

print('Registros duplicados: ', df['review_comment_message'].duplicated(keep=False).sum())
print(df['review_comment_message'].str.len().describe())

print("\nSentenças de 6 caracteres")

print(df['review_comment_message'][df['review_comment_message'].str.len() == 6].head())
rating_counts = df['review_score'].value_counts().reset_index().sort_values('index').iloc[:,1].tolist()

rating_p = round(df['review_score'].value_counts(normalize=True).reset_index().sort_values('index').iloc[:,1] * 100, 1).apply(lambda x: '{} %'.format(x)).tolist()



fig, ax = plt.subplots(figsize=(12,7))

ax = sns.countplot(x=df['review_score'])

plt.xticks(fontsize=12)

plt.title('Distruição das Classes de Avaliação',fontsize=20)

ax.set_ylabel('')

ax.set_xlabel('Nota da Avaliação', size=15)

for i, v in enumerate(rating_counts):

    ax.text(i-0.2, v+400, rating_p[i], size=20)
# =============================================================================

# Balanceamento

# =============================================================================

pos = df[['review_comment_message', 'review_score']][df['review_score'] == 5]

neg = df[['review_comment_message', 'review_score']][df['review_score'] <= 2]



df_ml = pd.concat([pos, neg])

maping = {5: 'Positivo'

         ,2: 'Negativo'

         ,1: 'Negativo'}

df_ml = df_ml.replace(maping)



np.random.seed(seed=100)

pos = np.random.choice(df_ml['review_comment_message'][df_ml['review_score'] == 'Positivo'], 10000, replace=False)

neg = np.random.choice(df_ml['review_comment_message'][df_ml['review_score'] == 'Negativo'], 10000, replace=False)

select = np.concatenate([pos, neg])



df_ml = df_ml[df_ml['review_comment_message'].isin(select)].reset_index(drop=True)

df_ml = df_ml.sample(frac=1).reset_index(drop=True)



print(df_ml['review_score'].value_counts(normalize=True))

print("\n", df_ml.shape)
# =============================================================================

# Instalação do pacote em Português do Spacy

# =============================================================================

!python -m spacy download pt
# =============================================================================

# Classe para Preprocessamento de Texto

# =============================================================================

nlp = spacy.load('pt', disable=['parser', 'ner'])

stemmer = nltk.stem.RSLPStemmer()



class DataPrep:

            

    def __init__(self):

        print('DataPrep ready.')

        

    def remove_stopwords(self, texto):

        """ Função para remover stopwords e outras palavras predefinidas"""

        stop_words = [word for word in nlp.Defaults.stop_words]

                

        #remover = ['lojas', 'americanas', 'americana', 'blackfriday', 'black', 'friday']

        

        #stop_words.extend(remover)

        

        

        

        texto_limpo = " ".join([i for i in texto if i not in set(stop_words)])

        return texto_limpo

    

    def clean_text(self, texto):

        """ Função para aplicar a remoção de stopwords, caracteres não alfabéticos e outras palavras curtas"""

        df_corpus = []

        for i in range(len(texto)):

            df_c = re.sub('[^A-Za-záàâãéèêíïóôõöúçñÁÀÂÃÉÈÍÏÓÔÕÖÚÇÑ]', ' ', texto[i]).lower().split()

            df_corpus.append(df_c)

        df_corpus= pd.Series(df_corpus).apply(lambda x: ' '.join([w for w in x if len(w)>2]))

        corpus = [self.remove_stopwords(r.split()) for r in df_corpus]

        return corpus



    def lemmatization(self, texto):

        """ Função para extrair o lema das palavras"""

        global nlp        

        output = []

        for sent in texto:

            doc = nlp(" ".join(sent)) 

            output.append([token.lemma_ for token in doc])

        return output



    def lemmatize(self, texto):

        """ Função para aplicar a limpeza do texto e a lemmatização"""

        token = self.lemmatization(pd.Series(self.clean_text(texto)).apply(lambda x: x.split()))

        token_lemma = []

        for i in range(len(token)):

            token_lemma.append(' '.join(token[i]))

        return token_lemma

    

    def list_freq(self, texto, terms=30):

        """ Função para listar palavras mais frequentes"""

        all_words = ' '.join([text for text in texto])

        all_words = all_words.split()

        fdist = FreqDist(all_words)

        words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

        d = words_df.nlargest(columns="count", n=terms) 

        return d

        print(d[:terms])

           

    def steming(self, texto):

        """ Função extrair a raiz (stem) das palavras"""

        global stemmer

        output_1 = []

        for sent in texto:

            doc = " ".join(sent)

            output_2 = []

            for w in doc.split():

                f = stemmer.stem(w)  

                output_2.append(f)

            output_1.append(output_2)    

        return output_1

        

    def stemize(self, texto):

        """ Função para aplicar o steming"""

        token = self.steming(pd.Series(self.clean_text(texto)).apply(lambda x: x.split()))

        token_lemma = []

        for i in range(len(token)):

            token_lemma.append(' '.join(token[i]))

        return token_lemma 

    

    def rm_accents(self, texto) -> list:

        '''A função irá remover acentos e cedilha'''

        fixed = list()

        for linha in texto:

            unidecoded_text = unidecode(linha)

            fixed.append(unidecoded_text)

        return fixed
%%time

# =============================================================================

# Pré-processamento do corpus

# =============================================================================

dp = DataPrep()



df_ml['review_comment_message'] = dp.rm_accents(df_ml['review_comment_message'])

corpus = dp.lemmatize(df_ml['review_comment_message'])
# =============================================================================

# Bag of Words

# =============================================================================

from PIL import Image

from wordcloud import WordCloud



word = ' '.join(corpus)



wc = WordCloud(

    background_color='black',

    max_words=2000

)

wc.generate(word)



fig = plt.subplots(figsize=(8,8))

plt.imshow(wc, interpolation='bilinear')

plt.axis('off')

plt.show()
%%time

# =============================================================================

# Vetorização TF-IDF

# =============================================================================

vectorizer = TfidfVectorizer(min_df=2, max_df=0.75, analyzer='word',

                             strip_accents='unicode', use_idf=True,

                             ngram_range=(1,2), max_features=10000)



X = vectorizer.fit_transform(corpus).toarray()

y = df_ml.loc[:,"review_score"].values



print('X shape: ', X.shape, '\ny shape: ', y.shape)
feature_names = vectorizer.get_feature_names()

feature_names[:20]
%%time

# =============================================================================

# Splitting the dataset into the Training set and Test set

# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# =============================================================================

# Fitting Naive Bayes to the Training set

# =============================================================================

classifier_nb = MultinomialNB()

classifier_nb.fit(X_train, y_train)

# =============================================================================

# Predicting the Test set results

# =============================================================================

y_pred = classifier_nb.predict(X_test)

y_pred_prob = classifier_nb.predict_proba(X_test)

# =============================================================================

# Model Evaluation

# =============================================================================

print('Acurácia de: ', "{0:.1f}".format(accuracy_score(y_test, y_pred)*100),'%')

print('Log Loss de: ', round(log_loss(y_test, y_pred_prob, eps=1e-15),4))
# =============================================================================

# Matriz de Confusão

# =============================================================================

cm = confusion_matrix(y_test, y_pred, labels=['Positivo', 'Negativo'])

'''

default:

[[TN, FP,

  FN, TP]]

  

labels = [1,0]:

[[TP, FP,

  FN, TN]]

'''



fig, ax = plt.subplots(figsize=(7,6))

sns.heatmap(cm, annot=True, ax=ax, fmt='.0f'); #annot=True to annotate cells



# labels, title and ticks

ax.set_xlabel('Predicted labels')

ax.set_ylabel('Real labels')

ax.set_title('Confusion Matrix')

ax.xaxis.set_ticklabels(['Positivo', 'Negativo'])

ax.yaxis.set_ticklabels(['Positivo', 'Negativo']);

plt.show()



sens = cm[0,0] / (cm[0,0] + cm[1,0])

esp = cm[1,1] / (cm[0,1] + cm[1,1])



print('Sensibilidade: ', round(sens,2))

print('Especificidade: ', round(esp,2))
import random

random.seed(5)

# =============================================================================

# Teste de entrada no modelo

# =============================================================================

testes = list()

for i in range(5):

    testes.append(random.choice(df_ml['review_comment_message']))



# =============================================================================

# Preparação dos dados

# =============================================================================

dp = DataPrep()

testes = dp.rm_accents(testes)

corpus2 = dp.lemmatize(testes)



# =============================================================================

# Resultados

# =============================================================================    

print("\nClassificação Predita:")

testes_transform = vectorizer.transform(corpus2)

for i in range(len(testes)):

    print("{} {:-<16} {}".format([i+1], classifier_nb.predict(testes_transform)[i], testes[i]))



print("\nProbabilidaes:")

testes_transform = vectorizer.transform(corpus2)

for i in range(len(testes)):

    print("{} {:-<16} {}".format([i+1], str([round(x,2) for x in classifier_nb.predict_proba(testes_transform)[i].tolist()]), testes[i]))
# =============================================================================

# Funções para Modelagem de Tópicos

# =============================================================================



# Coletar tópicos e seus pesos

def get_topics_terms_weights(weights, feature_names):

    feature_names = np.array(feature_names)

    sorted_indices = np.array([list(row[::-1]) for row in np.argsort(np.abs(weights))])

    sorted_weights = np.array([list(wt[index]) for wt, index in zip(weights, sorted_indices)])

    sorted_terms = np.array([list(feature_names[row]) for row in sorted_indices])

    

    topics = [np.vstack((terms.T, term_weights.T)).T for terms, term_weights in zip(sorted_terms, sorted_weights)]

    

    return topics





# Imprimir os componentes de cada tópico

def print_topics_udf(topics, total_topics=1,

                     weight_threshold=0.0001,

                     display_weights=False,

                     num_terms=None):

    

    for index in range(total_topics):

        topic = topics[index]

        topic = [(term, float(wt))

                 for term, wt in topic]

        #print(topic)

        topic = [(word, round(wt,2)) 

                 for word, wt in topic 

                 if abs(wt) >= weight_threshold]

                     

        if display_weights:

            print('Topic #'+str(index)+' with weights')

            print(topic[:num_terms]) if num_terms else topic

        else:

            print('Topic #'+str(index+1)+' without weights')

            tw = [term for term, wt in topic]

            print(tw[:num_terms]) if num_terms else tw



def get_topics_udf(topics, total_topics=1,

                     weight_threshold=0.0001,

                     num_terms=None):

    

    topic_terms = []

    

    for index in range(total_topics):

        topic = topics[index]

        topic = [(term, float(wt))

                 for term, wt in topic]

        #print(topic)

        topic = [(word, round(wt,2)) 

                 for word, wt in topic 

                 if abs(wt) >= weight_threshold]

        

        topic_terms.append(topic[:num_terms] if num_terms else topic)



    return topic_terms
%%time

# =============================================================================

# Non-Negative Matrix Factorization (NMF)

# =============================================================================

'''

https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

sklearn.decomposition.NMF(n_components=None, init=None, solver='cd', beta_loss='frobenius', 

                          tol=0.0001, max_iter=200, random_state=None, alpha=0.0, l1_ratio=0.0, 

                          verbose=0, shuffle=False)[source]

'''

from sklearn.decomposition import NMF

num_topics = 7



nmf = NMF(n_components=num_topics, random_state=42, alpha=0.1, l1_ratio=0.5, init='nndsvd')

nmf.fit(X)



nmf_weights = nmf.components_

nmf_feature_names = vectorizer.get_feature_names()



print(nmf)
# =============================================================================

# Tópicos e pesos de suas principais palavras 

# =============================================================================

topics = get_topics_terms_weights(nmf_weights, nmf_feature_names)

print_topics_udf(topics, total_topics=num_topics, num_terms=5, display_weights=True)
# =============================================================================

# Transformação e inserção dos tópicos no Dataset

# =============================================================================

topic_values = nmf.transform(X)

df_ml['topic'] = topic_values.argmax(axis=1)



labels = {0:'Entrega'

         ,1:'Recebimento do Produto'

         ,2:'Entrega'

         ,3:'Qualidade do Produto'

         ,4:'Recomendação da Loja/Vendedor' 

         ,5:'Entrega' 

         ,6:'Entrega'

}



df_ml = df_ml.replace(labels)

np.random.seed(seed=2)

df_ml.sample(15)
# =============================================================================

# Distribuição Geral

# =============================================================================

print("Distribuição geral do corpus:\n", df_ml['topic'].value_counts(normalize=True))



# =============================================================================

# Distribuição Avaliações Negativas

# =============================================================================

print("\nDistribuição das avaliações Negativas:\n", df_ml['topic'][df_ml['review_score'] == 'Negativo'].value_counts(normalize=True))



# =============================================================================

# Distribuição Avaliações Positivas

# =============================================================================

print("\nDistribuição das avaliações Positivas:\n", df_ml['topic'][df_ml['review_score'] == 'Positivo'].value_counts(normalize=True))
%%time

sent = [line.split() for line in corpus]

phrases = Phrases(sent, min_count=1, threshold=2, progress_per=1000) 

bigram = Phraser(phrases)

sentences = bigram[sent]
# =============================================================================

# Construindo o Modelo

# =============================================================================

w2v_model = Word2Vec(min_count=20

                    ,window=3

                    ,sg=0

                    ,size=100

                    ,sample=6e-5

                    ,alpha=0.03

                    ,min_alpha=0.0007

                    ,negative=20

                    ,workers=7

                    ,seed=42)
%%time

w2v_model.build_vocab(sentences, progress_per=100)
%%time

w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
w2v_model.init_sims(replace=True)

print("Model has %d terms" % len(w2v_model.wv.vocab))
# =============================================================================

# Termos mais similares

# =============================================================================

w2v_model.wv.most_similar(positive=["recomendar"])
# =============================================================================

# Similaridade entre dois termos

# =============================================================================

w2v_model.wv.similarity("comprar", 'atrasar')
from sklearn.manifold import TSNE

def display_closestwords_tsnescatterplot(model, word, n):

    

    arr = np.empty((0,100), dtype='f')

    word_labels = [word]



    # get close words

    close_words = model.wv.similar_by_word(word, topn=n)

    

    # add the vector for each of the closest words to the array

    arr = np.append(arr, np.array([model[word]]), axis=0)

    for wrd_score in close_words:

        wrd_vector = model[wrd_score[0]]

        word_labels.append(wrd_score[0])

        arr = np.append(arr, np.array([wrd_vector]), axis=0)

        

    # find tsne coords for 2 dimensions

    tsne = TSNE(n_components=2, random_state=0)

    np.set_printoptions(suppress=True)

    Y = tsne.fit_transform(arr)

    

    fig = plt.figure(figsize=(15,15))

    x_coords = Y[:, 0]

    y_coords = Y[:, 1]

    

    # display scatter plot

    plt.scatter(x_coords, y_coords)

    

    k=1

    for label, x, y in zip(word_labels, x_coords, y_coords):

        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')

    plt.xlim(x_coords.min()-k, x_coords.max()+k)

    plt.ylim(y_coords.min()-k, y_coords.max()+k)

    plt.show()
display_closestwords_tsnescatterplot(w2v_model, 'entregar', 100)