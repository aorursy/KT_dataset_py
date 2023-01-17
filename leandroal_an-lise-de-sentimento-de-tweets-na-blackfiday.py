'''



from bs4 import BeautifulSoup

import re

import time

from csv import DictWriter

import pprint

import datetime

from datetime import date, timedelta

from selenium import webdriver

from selenium.webdriver.common.by import By

from selenium.webdriver.support.ui import WebDriverWait

from selenium.webdriver.support import expected_conditions as EC

from selenium.common.exceptions import TimeoutException





def init_driver(driver_type):

    if driver_type == 1:

        driver = webdriver.Firefox()

    elif driver_type == 2:

        driver = webdriver.Chrome('/usr/bin/chromedriver')

    elif driver_type == 3:

        driver = webdriver.Ie()

    elif driver_type == 4:

        driver = webdriver.Opera()

    elif driver_type == 5:

        driver = webdriver.PhantomJS()

    driver.wait = WebDriverWait(driver, 5)

    return driver





def scroll(driver, start_date, end_date, words, lang, max_time):

    languages = { 1: 'en', 2: 'pt', 3: 'es', 4: 'fr', 5: 'de', 6: 'ru', 7: 'zh'}

    url = "https://twitter.com/search?f=tweets&vertical=default&q="

    for w in words[:-1]:

        url += "{}%20OR".format(w)

    url += "{}%20".format(words[-1])

    url += "since%3A{}%20until%3A{}&".format(start_date, end_date)

    if lang != 0:

        url += "l={}&".format(languages[lang])

    url += "src=typd"

    url += '&f=live'

    print(url)

    driver.get(url)

    start_time = time.time()  # remember when we started

    while (time.time() - start_time) < max_time:

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        print('Time left: ',  str(round(max_time - (time.time() - start_time))))





def scrape_tweets(driver):

    try:

        tweet_divs = driver.page_source

        obj = BeautifulSoup(tweet_divs, "html.parser")

        content = obj.find_all("div", class_="content")

        print(content)



        print("content printed")

        print(len(content))

        for c in content:

            tweet = c.find("p", class_="tweet-text").text.strip()

            tweet_text = re.sub('\n', ' ', tweet)

            print(tweet_text)

            print("-----------")

            try:

                name = (c.find_all("strong", class_="fullname")[0].text).strip()

            except AttributeError:

                name = "Anônimo"

            date = (c.find_all("span", class_="_timestamp")[0].text).strip()



            try:

                write_csv(date,tweet_text,name)

            except:

                print('csv error')



    except Exception as e:

        print("Something went wrong!")

        print(e)

        driver.quit()





def write_csv_header():

    with open("twitterData.csv", "w+", encoding='utf-8') as csv_file:

        fieldnames = ['Date', 'Name', 'Tweet']

        writer = DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()



def write_csv(date,tweet,name):

    with open("twitterData.csv", "a+", encoding='utf-8') as csv_file:

        fieldnames = ['Date', 'Name', 'Tweet']

        writer = DictWriter(csv_file, fieldnames=fieldnames)

        #writer.writeheader()

        writer.writerow({'Date': date,'Name': name,'Tweet': tweet})







def make_csv(data):

    l = len(data['date'])

    print("count: %d" % l)

    with open("twitterData.csv", "a+", encoding='utf-8') as file:

        fieldnames = ['Date', 'Name', 'Tweet']

        writer = DictWriter(file, fieldnames=fieldnames)

        writer.writeheader()

        for i in range(l):

            writer.writerow({'Date': data['date'][i],

                            'Name': data['name'][i],

                            'Tweet': data['tweet'][i]

                            })





def get_all_dates(start_date, end_date):

    dates = []

    start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")

    end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")

    step = timedelta(days=1)

    while start_date <= end_date:

        dates.append(str(start_date.date()))

        start_date += step



    return dates





def main():

    driver_type = 2

    wordsToSearch = ['blackfriday']

    for w in wordsToSearch:

        w = w.strip()

    start_date = '2019-11-15'

    end_date = '2019-12-16'

    lang = 2

    all_dates = get_all_dates(start_date, end_date)

    max_time = 600

    write_csv_header()

    for i in range(len(all_dates) - 1):

        driver = init_driver(driver_type)

        scroll(driver, str(all_dates[i]), str(all_dates[i + 1]), wordsToSearch, lang, max_time)

        scrape_tweets(driver)

        time.sleep(5)

        print("The tweets for {} are ready!".format(all_dates[i]))

        driver.quit()





if __name__ == "__main__":

    main()



'''
import pandas as pd

pd.set_option('max_colwidth', 400)

import numpy as np

import nltk

from nltk import FreqDist

import spacy

import re

from unidecode import unidecode



import pandas as pd

from gensim.models.phrases import Phrases, Phraser

from gensim.models import Word2Vec

import multiprocessing



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

#plt.style.use('seaborn')

import seaborn as sns

%matplotlib inline



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score, log_loss,confusion_matrix



#from nltk.corpus import stopwords

#from nltk.stem.rslp import RSLPStemmer

import  warnings

warnings.filterwarnings("ignore")
%%time

file = "/kaggle/input/b2wreviews/B2W-Reviews01.csv"

b2w_raw = pd.read_csv(file, sep=';', encoding='utf-8', low_memory=False)

b2w = b2w_raw[['review_text', 'overall_rating']]



b2w.info()
b2w['review_text'].str.len().describe()
# =============================================================================

# Textos Vazios

# =============================================================================

print('Avaliações nulas: ', b2w['review_text'].isnull().sum())



# =============================================================================

# Registros duplicados

# =============================================================================

print('Registros duplicados: ', b2w.duplicated(keep=False).sum())



# =============================================================================

# Formato Final

# =============================================================================

b2w = b2w.drop_duplicates().reset_index(drop=True)

print('Formato do dataset após remoção de duplicados:', b2w.shape)
rating_counts = b2w['overall_rating'].value_counts().reset_index().sort_values('index').iloc[:,1].tolist()

rating_p = round(b2w['overall_rating'].value_counts(normalize=True).reset_index().sort_values('index').iloc[:,1] * 100, 1).apply(lambda x: '{} %'.format(x)).tolist()



fig, ax = plt.subplots(figsize=(12,7))

ax = sns.countplot(x=b2w['overall_rating'])

plt.xticks(fontsize=12)

plt.title('Distruição das Classes de Avaliação',fontsize=20)

ax.set_ylabel('')

ax.set_xlabel('Nota da Avaliação', size=15)

for i, v in enumerate(rating_counts):

    ax.text(i-0.2, v+400, rating_p[i], size=20)
print('Distribuição Quantitativa:\n', b2w['overall_rating'].value_counts())



print('\nDistribuição Percentual:\n', b2w['overall_rating'].value_counts(normalize=True))
# =============================================================================

# Balanceamento

# =============================================================================

pos = b2w[['review_text', 'overall_rating']][b2w['overall_rating'] == 5]

neg = b2w[['review_text', 'overall_rating']][b2w['overall_rating'] <= 2]



b2w_ml = pd.concat([pos, neg])

maping = {5: 'Positivo'

         ,2: 'Negativo'

         ,1: 'Negativo'}

b2w_ml = b2w_ml.replace(maping)



np.random.seed(seed=100)

pos = np.random.choice(b2w_ml['review_text'][b2w_ml['overall_rating'] == 'Positivo'], 20000, replace=False)

neg = np.random.choice(b2w_ml['review_text'][b2w_ml['overall_rating'] == 'Negativo'], 20000, replace=False)

select = np.concatenate([pos, neg])



b2w_ml = b2w_ml[b2w_ml['review_text'].isin(select)].reset_index(drop=True)

b2w_ml = b2w_ml.sample(frac=1).reset_index(drop=True)



print(b2w_ml['overall_rating'].value_counts(normalize=True))

print("\n", b2w_ml.shape)
# =============================================================================

# Instalação do pacote em Português do Spacy

# =============================================================================

!python -m spacy download pt
nlp = spacy.load('pt', disable=['parser', 'ner'])

stemmer = nltk.stem.RSLPStemmer()



class DataPrep:

            

    def __init__(self):

        print('DataPrep ready.')

        

    def remove_stopwords(self, texto):

        stop_words = [word for word in nlp.Defaults.stop_words]

        #stop_words = nltk.corpus.stopwords.words('portuguese')

        

        remover = ['lojas', 'americanas', 'americana', 'blackfriday', 'black', 'friday']

        

        stop_words.extend(remover)

        

        """ Função para remover stopwords e outras palavras predefinidas"""

        

        texto_limpo = " ".join([i for i in texto if i not in set(stop_words)])

        return texto_limpo

    

    def clean_text(self, texto):

        df_corpus = []

        for i in range(len(texto)):

            df_c = re.sub('[^A-Za-záàâãéèêíïóôõöúçñÁÀÂÃÉÈÍÏÓÔÕÖÚÇÑ]', ' ', texto[i]).lower().split()

            df_corpus.append(df_c)

        df_corpus= pd.Series(df_corpus).apply(lambda x: ' '.join([w for w in x if len(w)>2]))

        corpus = [self.remove_stopwords(r.split()) for r in df_corpus]

        return corpus



    def lemmatization(self, texto):

        global nlp        

        output = []

        for sent in texto:

            doc = nlp(" ".join(sent)) 

            output.append([token.lemma_ for token in doc])

        return output



    def lemmatize(self, texto):

        token = self.lemmatization(pd.Series(self.clean_text(texto)).apply(lambda x: x.split()))

        token_lemma = []

        for i in range(len(token)):

            token_lemma.append(' '.join(token[i]))

        return token_lemma

    

    def list_freq(self, texto, terms=30):

        all_words = ' '.join([text for text in texto])

        all_words = all_words.split()

        fdist = FreqDist(all_words)

        words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())})

        d = words_df.nlargest(columns="count", n=terms) 

        return d

        print(d[:terms])

           

    def steming(self, texto):

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

        token = self.steming(pd.Series(self.clean_text(texto)).apply(lambda x: x.split()))

        token_lemma = []

        for i in range(len(token)):

            token_lemma.append(' '.join(token[i]))

        return token_lemma 

    

    def rm_accents(self, texto) -> list:

        '''

        A função irá remover acentos e cedilha.

        

        :param df: Dataframe a ser modificado 

        :param col: Coluna de entrada que contém os nomes a serem transformados

        :return: Lista de nomes transformados

        '''

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



b2w_ml['review_text'] = dp.rm_accents(b2w_ml['review_text'])

corpus = dp.lemmatize(b2w_ml['review_text'])
%%time

vectorizer = TfidfVectorizer(min_df=2, max_df=0.75, analyzer='word',

                             strip_accents='unicode', use_idf=True,

                             ngram_range=(1,1), max_features=15000)



X = vectorizer.fit_transform(corpus).toarray()

y = b2w_ml.iloc[:,-1].values



print('X shape: ', X.shape, '\ny shape: ', y.shape)
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
# =============================================================================

# Teste de entrada no modelo

# =============================================================================

testes = ["Excelente produto, recomendo"

         ,"Nao recebi o produto, quero meu dinheiro de volta"

         ,"não consigo encontrar maquiagem na black friday que meu bolso consegue pagar... triste & sem dinheiro"

         ,"Man tô na mesma situação tô com 100 reais na carteira da psn já a 2 semanas esperando a Black Friday pra comprar o days gone" # Padrão Twitter, modelo não conhece

         ]



# =============================================================================

# Preparação dos dados

# =============================================================================

dp = DataPrep()

testes = dp.rm_accents(testes)

corpus2 = dp.lemmatize(testes)



print("\nClassificacao Predita:")

testes_transform = vectorizer.transform(corpus2)

for i in range(len(testes)):

    print("{} {:-<16} {}".format([i+1], classifier_nb.predict(testes_transform)[i], testes[i]))



print("\nProbabilidaes:")

testes_transform = vectorizer.transform(corpus2)

for i in range(len(testes)):

    print("{} {:-<16} {}".format([i+1], str([round(x,2) for x in classifier_nb.predict_proba(testes_transform)[i].tolist()]), testes[i]))
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

# Non-Negative Matrix Factorization (NMF))

# =============================================================================

'''

https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html

sklearn.decomposition.NMF(n_components=None, init=None, solver='cd', beta_loss='frobenius', 

                          tol=0.0001, max_iter=200, random_state=None, alpha=0.0, l1_ratio=0.0, 

                          verbose=0, shuffle=False)[source]

'''

from sklearn.decomposition import NMF

num_topics = 10



nmf = NMF(n_components=num_topics, random_state=42, alpha=0.1, l1_ratio=0.5, init='nndsvd')

nmf.fit(X)



nmf_weights = nmf.components_

nmf_feature_names = vectorizer.get_feature_names()



print(nmf)
topics = get_topics_terms_weights(nmf_weights, nmf_feature_names)

print_topics_udf(topics, total_topics=num_topics, num_terms=5, display_weights=True)
# =============================================================================

# Transformação e inserção dos tópicos no Dataset

# =============================================================================

topic_values = nmf.transform(X)

b2w_ml['topic'] = topic_values.argmax(axis=1)



labels = {0:'Qualidade do Produto'

         ,1:'Recebimento do Produto'

         ,2:'Prazo de Entrega'

         ,3:'Prazo de Entrega'

         ,4:'Qualidade do Produto' 

         ,5:'Qualidade do Produto' 

         ,6:'Qualidade do Produto'

         ,7:'Qualidade do Produto'

         ,8:'Qualidade do Produto'

         ,9:'Qualidade do Produto'

}



b2w_ml = b2w_ml.replace(labels)

b2w_ml.head(10)
# =============================================================================

# Distribuição Geral

# =============================================================================

print("Distribuição geral do corpus:\n", b2w_ml['topic'].value_counts(normalize=True))



# =============================================================================

# Distribuição Avaliações Negativas

# =============================================================================

print("\nDistribuição das avaliações Negativas:\n", b2w_ml['topic'][b2w_ml['overall_rating'] == 'Negativo'].value_counts(normalize=True))
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

# Termos mais antagônicos

# =============================================================================

w2v_model.wv.most_similar(negative=["recomendar"])
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
display_closestwords_tsnescatterplot(w2v_model, 'comprar', 100)
file = "/kaggle/input/twitter-black-friday/twitterData.csv"

twitter_df = pd.read_csv(file)

print('twitter_df shape: ', twitter_df.shape)

twitter_df.head()
def rm_spam(df, col):

    index_list = list()

    for i in range(df.shape[0]):

        match = re.search(r"\Shttp|\bhttp", df[col][i])

        if match:

            index_list.append(i)

        else:

            continue

    df_filtered = df.drop(index_list, axis=0).reset_index(drop=True)

    return df_filtered



twitter_df = rm_spam(twitter_df, col='Tweet')

twitter_df = twitter_df.drop_duplicates('Tweet').reset_index(drop=True)

print('twitter_df shape: ', twitter_df.shape)

twitter_df.head()
twitter_df['Year'] = '2019'

twitter_df['Month'] = np.where(twitter_df['Date'].str[:3] == 'Nov', '11','12')

twitter_df['Day'] = twitter_df['Date'].str[-2:]

twitter_df = twitter_df[~twitter_df['Day'].str.contains(r'h$|m$')]

twitter_df['Day'] = twitter_df['Day'].str.strip().astype(int)



new_day = []

for line in twitter_df['Day']:

    line = '{:02d}'.format(line)

    new_day.append(line)

    

twitter_df['Day'] = new_day



twitter_df['DateNew'] = pd.to_datetime(twitter_df['Year']+twitter_df['Month']+twitter_df['Day'].apply(str))
twitter_plot = twitter_df.copy()

twitter_plot['DateNew'] = twitter_plot['DateNew'].dt.date

twitter_plot = twitter_plot.groupby('DateNew')['Tweet'].count()



fig, ax = plt.subplots(figsize=(15,5))

ax = twitter_plot.plot(kind='bar')

plt.xticks(fontsize=14, rotation=30)

plt.xlabel('Data')



plt.show()
twitter_df_filtered = twitter_df[(twitter_df['DateNew'] >= pd.to_datetime('20191128')) & (twitter_df['DateNew'] <= pd.to_datetime('20191201'))]

twitter_df_filtered = twitter_df_filtered[['DateNew', 'Tweet']].reset_index(drop=True)



print('Dimensão dos Dados Filtrados: ', twitter_df_filtered.shape)
%%time

# =============================================================================

# Pré-processamento do corpus

# =============================================================================

dp = DataPrep()

corpus_twt = dp.rm_accents(twitter_df_filtered['Tweet'])

corpus_twt = dp.lemmatize(corpus_twt)
testes_vec = vectorizer.transform(corpus_twt)

for i in range(5):

    print("{} {:-<16} {}".format([i+1], classifier_nb.predict(testes_vec)[i], twitter_df_filtered.iloc[i,1]))
# =============================================================================

# Dados de classificação

# =============================================================================

testes_pred = classifier_nb.predict(testes_vec)



# =============================================================================

# Dados de probabilidade de "Positivo"

# =============================================================================

testes_probs = classifier_nb.predict_proba(testes_vec)



# =============================================================================

# Inserção dos registros no Dataset

# =============================================================================

twitter_df_filtered['Predicted'] = testes_pred

    

twitter_df_filtered['Positive_Prob']  = 0

for i in range(twitter_df_filtered.shape[0]):

    twitter_df_filtered['Positive_Prob'].iloc[i] = testes_probs[i][1]

    

twitter_df_filtered
# =============================================================================

# Dados mais assertivos

# =============================================================================

selection = twitter_df_filtered[(twitter_df_filtered['Positive_Prob'] >= 0.9) | (twitter_df_filtered['Positive_Prob'] <= 0.1)].reset_index(drop=True)

print('Formato do dataset:', selection.shape)

selection.head(10)
selection[selection['Positive_Prob'] > 0.9]
selection['Predicted'].value_counts(normalize=True)
selection.to_csv("selectedPreds.csv", index=None)