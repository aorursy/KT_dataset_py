import pandas as pd



df = pd.read_csv('../input/coronavirus-uk-newspapers/Newspaper_links.csv',encoding='utf-8')

print('The dataframe has %i rows and %i columns'%(df.shape[0],df.shape[1]))

df.head()
import numpy as np



papers = np.unique(df['Paper'])

print('There are %i different newspapers in the database.'%len(papers))

print('They are %s.\n'%', '.join(papers))

for p in papers:

        print('There are %i links for the %s'%(np.sum(df['Paper']==p),p))
import requests

from bs4 import BeautifulSoup

import re



guardian = df[df['Paper']=='guardian'].reset_index()

for ii in range(3):

    soup = BeautifulSoup(requests.get(guardian.loc[ii,'Link']).content, 'html5lib')

    Paragraphs = soup.find_all('p')

    for p in Paragraphs:

            print(p)

    print('\n\n\n\n')
mail = df[df['Paper']=='mail'].reset_index()

for ii in range(3):

    soup = BeautifulSoup(requests.get(mail.loc[ii,'Link']).content, 'html5lib')

    Paragraphs = soup.find_all('p')

    for p in Paragraphs:

            print(p)

    print('\n\n\n\n')
metro = df[df['Paper']=='metro'].reset_index()

for ii in range(3):

    soup = BeautifulSoup(requests.get(metro.loc[ii,'Link']).content, 'html5lib')

    Paragraphs = soup.find_all('p')

    for p in Paragraphs:

            print(p)

    print('\n\n\n\n')
sun = df[df['Paper']=='sun'].reset_index()

for ii in range(3):

    soup = BeautifulSoup(requests.get(sun.loc[ii,'Link']).content, 'html5lib')

    Paragraphs = soup.find_all('p')

    for p in Paragraphs:

            print(p)

    print('\n\n\n\n')
telegraph = df[df['Paper']=='telegraph'].reset_index()

for ii in range(3):

    soup = BeautifulSoup(requests.get(telegraph.loc[ii,'Link']).content, 'html5lib')

    Paragraphs = soup.find_all('p')

    for p in Paragraphs:

            print(p)

    print('\n\n\n\n')
times = df[df['Paper']=='times'].reset_index()

for ii in range(3):

    soup = BeautifulSoup(requests.get(times.loc[ii,'Link']).content, 'html5lib')

    Paragraphs = soup.find_all('p')

    for p in Paragraphs:

            print(p)

    print('\n\n\n\n')
import re



def format_newspaper(S):

    a = re.sub('\w*@\w*','',re.sub('<.+?>','',S)).replace('.co.uk','').replace('.com','').replace('\n','')

    a = re.sub('[^a-z 0-9]', '', a.lower()).replace(paper,'').replace('-','') + ' '

    return a    
import pyprind



prog = pyprind.ProgBar(df.shape[0])

Paragraph = []

for ii in range(df.shape[0]):

    url = df.loc[ii,'Link']

    try:

        soup = BeautifulSoup(requests.get(url).content, 'html5lib')

        title = str(soup.find('title'))

        if 'not found' in title:

            Paragraph.append('')

        else:

            paper = df.loc[ii,'Paper']

            full = ''

            Para = soup.find_all('p')

            len_para = len(Para)

            if len_para>2:

                for idx, p in enumerate(Para):

                    p_str = str(p)

                    if ((paper == 'sun' and idx<len_para-2) or paper in ('metro','guardian')) and p_str[:3]=='<p>':

                        full += format_newspaper(p_str)

                    elif paper == 'mail' and 'mol-para-with-font' in p_str:

                        full += format_newspaper(p_str)

                    elif paper == 'telegraph' and idx<len_para-6:

                        full += format_newspaper(p_str)

                    elif paper == 'times' and 'responsiveweb__Paragraph-sc-1isfdlb-0 YieBL' in p_str:

                        full += format_newspaper(p_str)

            Paragraph.append(full)

    except:

        Paragraph.append('')

    prog.update()

df['Text'] = Paragraph

df=df[['Paper','Link','ExtractDatetime','Text']]

df = df[df['Text']!='']
for p in papers:

        print('There are %i successfully extracted news articles for the %s'%(np.sum(df['Paper']==p),p))
def tokenizer(text):

    return text.split()
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

X = df['Text']

y = le.fit_transform(df['Paper'])
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import make_pipeline



tfidf = TfidfVectorizer(strip_accents = None

                        , lowercase = False

                        , preprocessor = None

                        , stop_words = None

                        , ngram_range = (1,2)

                        , tokenizer = tokenizer)

lr = LogisticRegression(solver = 'liblinear'

                       ,class_weight='balanced'

                       ,max_iter = 10000

                       ,penalty = 'l1'

                       ,C = 10.0)

pipe = make_pipeline(tfidf, lr)
##Logistic Regression pipeline parameter space:

# param_grid = [

#                {

#                    'vect__ngram_range':[(1,1),(1,2),(1,3),(1,4),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)]

#                    ,'vect__tokenizer':[tokenizer, tokenizer_porter]

#                     ,'vect__stop_words':[stop,None]

#                    ,'clf__penalty':['l1','l2']

#                    ,'clf__C':[0.01, 0.1, 1.0, 10.0, 100.0]

#                }

#            ]



##Support vector pipeline parameter space:

# param_grid = [

#                {

#                    'vect__ngram_range':[(1,1),(1,2),(1,3),(1,4),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)]

#                    ,'vect__tokenizer':[tokenizer, tokenizer_porter]

#                     ,'vect__stop_words':[stop,None]

#                    ,'clf__kernel':['linear']

#                    ,'clf__C':[0.01, 0.1, 1.0, 10.0, 100.0]

#                }

#                ,{

#                    'vect__ngram_range':[(1,1),(1,2),(1,3),(1,4),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)]

#                    ,'vect__tokenizer':[tokenizer, tokenizer_porter]

#                     ,'vect__stop_words':[stop,None]

#                    ,'clf__kernel':['rbf']

#                    ,'clf__gamma':['scale','auto', 0.1, 1.0, 10.0]

#                    ,'clf__C':[0.01, 0.1, 1.0, 10.0, 100.0]

#                }

#            ]



##Random forest pipeline parameter space:

# param_grid = [

#                {

#                    'vect__ngram_range':[(1,1),(1,2),(1,3),(1,4),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)]

#                    ,'vect__tokenizer':[tokenizer, tokenizer_porter]

#                     ,'vect__stop_words':[stop,None]

#                    ,'clf__n_estimators' : [100,250,500,750,1000]

#                    ,'clf__max_depth' : [1,None]

#                }

#            ]



##Naive Bayes pipeline parameter space:

# param_grid = [

#                {

#                    'vect__ngram_range':[(1,1),(1,2),(1,3),(1,4),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7)]

#                    ,'vect__tokenizer':[tokenizer, tokenizer_porter]

#                     ,'vect__stop_words':[stop,None]

#                    ,'clf__alpha' : [0.0, 0.1, 0.5, 1.0]

#                    ,'clf__fit_prior' : [True,False]

#                }

#            ]
from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, stratify = y)



Model = pipe.fit(X_train,y_train)

print('The models accuracy on the training dataset is: %.3f'%Model.score(X_train,y_train))

print('The models accuracy on the testing  dataset is: %.3f'%Model.score(X_test,y_test))

papers = list(le.inverse_transform([0,1,2,3,4,5]))

print('The confusion matrix for the training dataset is:')

plot_confusion_matrix(Model

                      , X_train

                      , y_train

                      , display_labels = papers

                      , cmap=plt.cm.Blues

                      , values_format = '.5g')

plt.grid(False)

plt.show()

      

print('The confusion matrix for the testing dataset is:')

plot_confusion_matrix(Model

                      , X_test

                      , y_test

                      , display_labels = papers

                      , cmap=plt.cm.Blues

                      , values_format = '.5g')

plt.grid(False)

plt.show()
from sklearn.feature_extraction.text import CountVectorizer

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords



stop = stopwords.words('english')

count = CountVectorizer(strip_accents = None

                        , lowercase = False

                        , preprocessor = None

                        , tokenizer = tokenizer

                        , ngram_range = (1,2)

                        , stop_words = stop)



def COUNTVEC(X,y,num_of_words,idx):

    C = count.fit_transform(X[y==idx]).toarray()

    inv_vocab_count = {v:k for k,v in count.vocabulary_.items()}

    i = C.sum(axis = 0).argsort()[-num_of_words:][::-1]

    words = []

    for ii in range(num_of_words):

        words.append(inv_vocab_count[i[ii]])

    words = ', '.join(words)

    return words



num_of_words = 10

print('Most common non-stop-words tokens:\n')



for idx, p in enumerate(papers):

    print('\n')

    words = COUNTVEC(X_train,y_train, num_of_words, idx)

    print('The %s training data: '%p)

    print('\t%s'%words)

    

    words = COUNTVEC(X_test,y_test, num_of_words, idx)

    print('The %s testing data: '%p)

    print('\t%s'%words)

inv_vocab = {v:k for k,v in pipe[0].vocabulary_.items()}

num_of_words = 5



for idx, p in enumerate(papers):

    print('The %s:'%p)

    i = pipe[1].coef_[idx].argsort()[-num_of_words:][::-1]

    words = []

    for ii in range(num_of_words):

        words.append(inv_vocab[i[ii]])

    words = ', '.join(words)

    print('\tMost positively influential tokens: ')

    print('\t\t%s'%words)

    i = pipe[1].coef_[idx].argsort()[:num_of_words][::1]

    words = []

    for ii in range(num_of_words):

        if  pipe[1].coef_[idx][i[ii]]<0:

            words.append(inv_vocab[i[ii]])

        else:

            words.append(str(pipe[1].coef_[idx][i[ii]]))

    words = ', '.join(words)

    print('\tMost negatively influential tokens: ')

    print('\t\t%s'%words)    

    print('\n')