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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import re



import pickle 

#import mglearn

import time





from nltk.tokenize import TweetTokenizer # doesn't split at apostrophes

import nltk

from nltk import Text

from nltk.tokenize import regexp_tokenize

from nltk.tokenize import word_tokenize  

from nltk.tokenize import sent_tokenize 

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.stem import PorterStemmer





from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression 

from sklearn.naive_bayes import MultinomialNB

from sklearn.multiclass import OneVsRestClassifier





from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline



from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC
train_df = pd.read_csv("/kaggle/input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv")
train_df.head()
train_df['Genre length'] = train_df['Genre'].apply(len)

train_df.head()
train_df['Genre length'].describe()
train_df.describe()
train_df['Genre'].nunique()
genre_arr = train_df['Genre'].unique()

genre_data = train_df['Genre'].value_counts()

genre_data
type(genre_data)
import string

alpha = string.ascii_lowercase

alpha_up = string.ascii_uppercase

print(alpha)

print(alpha_up)
def remove_punc(data):

    string = ''

    for i in data:

        if i in alpha:

            string = string + i

        elif i in alpha_up:

            string = string + i

        elif i == ' ':

            string = string + i

        else:

            continue

    return string
dat = 'In a mansion called Xanadu, part of a vast palatial estate in Florida, the elderly Charles Foster Kane is on his deathbed. Holding a snow globe, he utters a word, "Rosebud", and dies; the globe slips from his hand and smashes on the floor'

print(remove_punc(dat))
def genre_corr(data):

    data = data.split()

    final_gen = ''

    list_rom = ['romance','love','love story','musical b', 'romantic','rom-coms','music','musical','actionlove','romanceaction',

               'romancecomedy','romancehorror','romcom','rom\|com','rom',' \(artistic\)',"drama|romance|adult|children"

               ]

    list_act = ['act','action','adventures','kung fu','martial arts','world war ii','world war i','spy film','biker film',

               'buddy cop','buddy film','bruceploitation','drama about child soldiers',"war-time","wartime","ww1","wwii",

               'true crime','crime','\|007','gun fu','afghan war drama','actionadventure','actioncomedy','actiondrama',

                'actionlove','actionmasala','actionchildren','adventurecomedy','actionthriller','martialarts',' \(volleyball\)',

                ' \(aquatics|swimming\)',' \(aquatics|swimming\)',' \(shogi|chess\)',' (road bicycle racing)','american football',

                'dev\|nusrat jahan',' \(road bicycle racing\)','liveaction','heistcomedy','heist','historydisaster','warcomedy',

                'samurai','martial_arts','adventure','spy','superhero',"drama|romance|adult|children",'actionner',

               ]

    list_sus = ['ttriller','coming of age','coming-of-age','slice of life','psycho thriller,',"ero",'actionadventure','dramathriller',

                'dramathriller','thriler','crimethriller','actionthriller','comedysocial','erotica','erotic','comedythriller',

                  'colour\|yellow\|productions\|eros\|international',       'melodrama', 'gangsterthriller',  'ancientcostume', 

                'dramatic','biodrama','bio-drama','comedy-drama adaptation of the mordecai richler novel','drama about child soldiers',

                'drama loosely','slice of life',"comedy–drama"'actionlove','actiondrama','fantasycomedy','dramacomedy',

                'dramacomedysocial','dramathriller','comedydrama','comedyhorror','adventurecomedy','animationdrama','comedysocial',

                'erotica','erotic','biblical','biblical','colour\|yellow\|productions\|eros\|international','liveaction','superheroes',

                'heistcomedy','heist','warcomedy','dramatic','familya','familya','dramedy','dramaa','famil\|','superheroe',

                'devotionalbiography','familydrama','espionage','romancefiction','horrorthriller','suspensethriller','triller',

                'satirical','homosexual','sexual','mockumentary','periodic','politics','tv_miniseries','serial',"musical–comedy",

                "roman|porno","action—masala","horror–thriller",'family','martial_arts','horror','war','adventure','noir',

                'superhero','social','suspense',"drama|romance|adult|children",'actionner',

                ]

    list_sci = ['animated','anime','children\'s','3-d','3d','sci-fi','sci fi','science fiction','avant-garde','animationchildren',

               'computer animation',   ' in animation',   'actionchildren',  'fantasychildren\|','fantasycomedy','fantasyperiod',

                'sciencefiction','animationdrama','fantay','\|\(children\|poker\|karuta\)','superheroes','computeranimation',

                '\|\(fiction\)','science_fictionchildren','science_fiction','superhero',"drama|romance|adult|children"

                ]

    list_hor = ['psychological','j-horror','psycho thriller,',"comedy–horror",'actionadventure','comedyhorror','horror',

                ]

    for i in data:

        if i.lower() in list_rom:

            final_gen = final_gen + ' Romance'

        elif i.lower() in list_act:

            final_gen = final_gen + ' Action'

        elif i.lower() in list_sus:

            final_gen = final_gen + ' Suspense'

        elif i.lower() in list_sci:    

            final_gen = final_gen + ' Science Fiction'

        elif i.lower() in list_hor:

            final_gen = final_gen + ' Horror'

        else:

            final_gen = final_gen + ' Others'

    final_gen = set(final_gen.split())

    genre = ''

    for i in final_gen:

        genre = genre + ' {}'.format(i)

    return genre

            
train_df['Genre String'] = train_df['Genre'].apply(remove_punc)

train_df.head(20)
train_df['Genre Corrected'] = train_df['Genre String'].apply(genre_corr)

train_df.head(20)
train_df['Genre Corrected 2'] = train_df['Genre'].apply(genre_corr)

train_df.head()
train_df.head(20)
train_df['Genre Corrected'].value_counts()
train_df['Genre Corrected 2'].value_counts()
train_df.columns
train_df.drop(['Genre Corrected 2','Genre length','Genre String',],axis=1,inplace=True)

train_df.head(20)
train_df.drop(['Genre'],axis=1,inplace=True)
train_df['GenreSplit']=train_df['Genre Corrected'].str.split()

train_df.head(20)
train_df.drop(['Genre Corrected'],axis=1,inplace=True)

train_df.head(20)


def text_process(word):

    """

    Takes in a string of text, then performs the following:

    1. Remove all punctuation

    2. Remove all stopwords

    3. Returns a list of the cleaned text

    """

    

    list_final = []

    ind = 0

    

    # Now just remove any stopwords

    for i in word.split():

        ind = ind + 1

        print(ind)

        if i.lower() not in stopwords.words('english'):

            list_final.append(i)

        else:

            continue

    

    return list_final
train_df['Plot Length'] = train_df['Plot'].apply(len)
train_df.head(20)
train_df['Plot Length'].sum()
for i in range(5):

    print(train_df['Plot'][i])
train_df['Plot without punc'] = train_df['Plot'].apply(remove_punc)
import nltk



def format_sentence(char):

    return({word: True for word in nltk.word_tokenize(char)})
train_df['Plot token'] = train_df['Plot'].apply(format_sentence)

train_df.head()
train_df['Plot final clean'] = train_df['Plot without punc'].apply(text_process)
train_df[train_df['Plot'].apply(len)== 582].index()