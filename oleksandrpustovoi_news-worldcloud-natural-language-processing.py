# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

from nltk.corpus import stopwords

from PIL import Image

from wordcloud import WordCloud,STOPWORDS

import matplotlib.pyplot as plt

from string import punctuation

import nltk

from nltk.corpus import stopwords



!pip install pymystem3

from pymystem3 import Mystem



!pip install pymorphy2

import pymorphy2





%matplotlib inline
df = pd.read_csv('/kaggle/input/economic-news-of-cyprus-ru/VesKipLinks.csv', index_col = 'Unnamed: 0')
print(f'Size {df.shape}')

df = df.drop('link', axis=1)

df = df.drop('hits', axis=1)

df.sample(2)
mystem = Mystem() 

nltk.download("stopwords")

stopwords = stopwords.words("russian") + stopwords.words("english") # some articles contain english quotes/parts



def preprocess_text(text):

    '''lemmatizes text'''

    text = str(text)

    words = mystem.lemmatize(text.lower())

    words = [word for word in words if word not in stopwords\

              and word != " " \

              and word.strip() not in punctuation]

    

    text = " ".join(words)

    

    return text
df['lemmatized_content'] = df.content.apply(preprocess_text)
def replacer (phrase):

    '''replaces hand-picked items'''

    new_phrase = ''

    for word in phrase.split(" "):

        new_word = word

        if new_word.lower().replace(' ', '') in ['кипра', 'кипр кипр' 'кипре', 'кипру', 'кипром', 'кипр','cyprus','кипрский','кипрa']:

            new_word = 'кипр'

        if new_word.lower().replace(' ', '') in ['банка', 'bank']:

            new_word = 'банк'

        if new_word.lower().replace(' ', '') in ['новое', 'new']:

            new_word = 'новый'

        if new_word.lower().replace(' ', '') in ['растить', 'вырастать', 'вырастить']:

            new_word = 'расти'

        new_phrase += new_word + ' '

    return new_phrase
df['lemmatized_content'] = df.lemmatized_content.apply(replacer)
def date_process(t):

    '''extracts year'''

    try:

        return(int(t[-10:-6]))

    except:

        pass

    

df.date = df.date.apply(date_process)
df.date.unique()
morph = pymorphy2.MorphAnalyzer()



list_1=['PREP','CONJ', 'PRCL', 'INTJ','NPRO']

def drop_grafems (phrase,list=list_1):

    new_phrase = ''

    for word in phrase.split(" "):

        new_word = morph.parse(word)[0]

        if not new_word.tag.POS in list:

            new_phrase += new_word.normal_form + ' '

    return new_phrase 







list_2=['NOUN', 'VERB', 'ADJF']

def filter_grafems (phrase,list=list_2):

    new_phrase = ''

    for word in phrase.split(" "):

        new_word = morph.parse(word)[0]

        if new_word.tag.POS in list:

            new_phrase += new_word.normal_form + ' '

    return new_phrase 







'''

Граммема	Значение	Примеры

* NOUN	имя существительное	хомяк

* ADJF	имя прилагательное (полное)	хороший

* ADJS	имя прилагательное (краткое)	хорош

* COMP	компаратив	лучше, получше, выше

* VERB	глагол (личная форма)	говорю, говорит, говорил

* INFN	глагол (инфинитив)	говорить, сказать

* PRTF	причастие (полное)	прочитавший, прочитанная

* PRTS	причастие (краткое)	прочитана

* GRND	деепричастие	прочитав, рассказывая

* NUMR	числительное	три, пятьдесят

* ADVB	наречие	круто

* NPRO	местоимение-существительное	он

* PRED	предикатив	некогда

* PREP	предлог	в

* CONJ	союз	и

* PRCL	частица	бы, же, лишь

* INTJ	междометие	ой

'''

pass
df['lemmatized_content_filtered'] = df.lemmatized_content.apply(filter_grafems)
#?WordCloud

# run above line to get doctring
mask = np.array(Image.open( "../input/pictures/cyprus.jpg"))



def wordcloud_draw(data, color = 'black'):

    '''draws a chart'''

    words = ' '.join(data)

    cleaned_word = " ".join([word for word in words.split()])

    wordcloud = WordCloud(stopwords=stopwords,

                      background_color=color, 

                          mask = mask,

                      width=640,

                      height=480,

                          max_words=350,

                          colormap="nipy_spectral",

                          scale=10,

                          ).generate(cleaned_word)

    wordcloud.to_file('N.png') # saving chart to file - just in case

    plt.figure(1,figsize=(25, 25))

    plt.imshow(wordcloud,  cmap=plt.cm.gray, interpolation="bicubic")

    plt.margins(x=0, y=0)



    plt.axis('off')

    plt.show()
wordcloud_draw(df.lemmatized_content_filtered,'white')