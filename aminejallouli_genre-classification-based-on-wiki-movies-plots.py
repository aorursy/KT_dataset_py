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
# 

#nRowsRead = 1000 # specify 'None' if want to read whole file

#movies = pd.read_csv('wiki_movie_plots_deduped.csv', delimiter=',', nrows = nRowsRead)

movies = pd.read_csv('../input/wiki_movie_plots_deduped.csv', delimiter=',')

movies.dataframeName = 'wiki_movie_plots_deduped.csv'

nRow, nCol = movies.shape

print(f'There are {nRow} rows and {nCol} columns')
movies.head()
# creation of the column count for aggregation

movies['Count']=1

movies[['Genre','Count']].groupby(['Genre'], as_index=False).count().shape[0]
# harmonization

movies['GenreCorrected'] =movies['Genre'] 

movies['GenreCorrected']=movies['GenreCorrected'].str.strip()

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' - ', '|')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' / ', '|')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('/', '|')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' & ', '|')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(', ', '|')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('; ', '|')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('bio-pic', 'biography')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biopic', 'biography')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biographical', 'biography')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biodrama', 'biography')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('bio-drama', 'biography')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biographic', 'biography')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(film genre\)', '')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('animated','animation')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('anime','animation')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('children\'s','children')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('comedey','comedy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\[not in citation given\]','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' set 4,000 years ago in the canadian arctic','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('historical','history')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('romantic','romance')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('3-d','animation')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('3d','animation')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('viacom 18 motion pictures','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('sci-fi','science_fiction')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('ttriller','thriller')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('.','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('based on radio serial','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' on the early years of hitler','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('sci fi','science_fiction')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('science fiction','science_fiction')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' (30min)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('16 mm film','short')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\[140\]','drama')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\[144\]','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' for ','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('adventures','adventure')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('kung fu','martial_arts')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('kung-fu','martial_arts')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('martial arts','martial_arts')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('world war ii','war')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('world war i','war')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biography about montreal canadiens star|maurice richard','biography')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('bholenath movies|cinekorn entertainment','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(volleyball\)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('spy film','spy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('anthology film','anthology')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biography fim','biography')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('avant-garde','avant_garde')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biker film','biker')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('buddy cop','buddy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('buddy film','buddy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('comedy 2-reeler','comedy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('films','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('film','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biography of pioneering american photographer eadweard muybridge','biography')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('british-german co-production','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('bruceploitation','martial_arts')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('comedy-drama adaptation of the mordecai richler novel','comedy-drama')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('movies by the mob\|knkspl','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('movies','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('movie','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('coming of age','coming_of_age')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('coming-of-age','coming_of_age')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('drama about child soldiers','drama')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('(( based).+)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('(( co-produced).+)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('(( adapted).+)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('(( about).+)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('musical b','musical')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('animationchildren','animation|children')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' period','period')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('drama loosely','drama')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(aquatics|swimming\)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(aquatics|swimming\)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace("yogesh dattatraya gosavi's directorial debut \[9\]",'')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace("war-time","war")

movies['GenreCorrected']=movies['GenreCorrected'].str.replace("wartime","war")

movies['GenreCorrected']=movies['GenreCorrected'].str.replace("ww1","war")

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('unknown','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace("wwii","war")

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('psychological','psycho')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('rom-coms','romance')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('true crime','crime')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|007','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('slice of life','slice_of_life')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('computer animation','animation')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('gun fu','martial_arts')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('j-horror','horror')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(shogi|chess\)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('afghan war drama','war drama')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|6 separate stories','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(30min\)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' (road bicycle racing)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' v-cinema','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('tv miniseries','tv_miniseries')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|docudrama','\|documentary|drama')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' in animation','|animation')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('((adaptation).+)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('((adaptated).+)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('((adapted).+)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('(( on ).+)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('american football','sports')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dev\|nusrat jahan','sports')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('television miniseries','tv_miniseries')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(artistic\)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \|direct-to-dvd','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('history dram','history drama')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('martial art','martial_arts')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('psycho thriller,','psycho thriller')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|1 girl\|3 suitors','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' \(road bicycle racing\)','')

filterE = movies['GenreCorrected']=="ero"

movies.loc[filterE,'GenreCorrected']="adult"

filterE = movies['GenreCorrected']=="music"

movies.loc[filterE,'GenreCorrected']="musical"

filterE = movies['GenreCorrected']=="-"

movies.loc[filterE,'GenreCorrected']=''

filterE = movies['GenreCorrected']=="comedy–drama"

movies.loc[filterE,'GenreCorrected'] = "comedy|drama"

filterE = movies['GenreCorrected']=="comedy–horror"

movies.loc[filterE,'GenreCorrected'] = "comedy|horror"

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(' ','|')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace(',','|')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('-','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actionadventure','action|adventure')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actioncomedy','action|comedy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actiondrama','action|drama')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actionlove','action|love')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actionmasala','action|masala')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actionchildren','action|children')



movies['GenreCorrected']=movies['GenreCorrected'].str.replace('fantasychildren\|','fantasy|children')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('fantasycomedy','fantasy|comedy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('fantasyperiod','fantasy|period')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('cbctv_miniseries','tv_miniseries')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dramacomedy','drama|comedy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dramacomedysocial','drama|comedy|social')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dramathriller','drama|thriller')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('comedydrama','comedy|drama')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dramathriller','drama|thriller')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('comedyhorror','comedy|horror')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('sciencefiction','science_fiction')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('adventurecomedy','adventure|comedy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('animationdrama','animation|drama')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|\|','|')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('muslim','religious')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('thriler','thriller')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('crimethriller','crime|thriller')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('fantay','fantasy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actionthriller','action|thriller')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('comedysocial','comedy|social')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('martialarts','martial_arts')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|\(children\|poker\|karuta\)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('epichistory','epic|history')



movies['GenreCorrected']=movies['GenreCorrected'].str.replace('erotica','adult')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('erotic','adult')



movies['GenreCorrected']=movies['GenreCorrected'].str.replace('((\|produced\|).+)','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('chanbara','chambara')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('comedythriller','comedy|thriller')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biblical','religious')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biblical','religious')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('colour\|yellow\|productions\|eros\|international','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|directtodvd','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('liveaction','live|action')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('melodrama','drama')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('superheroes','superheroe')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('gangsterthriller','gangster|thriller')



movies['GenreCorrected']=movies['GenreCorrected'].str.replace('heistcomedy','comedy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('heist','action')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('historic','history')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('historydisaster','history|disaster')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('warcomedy','war|comedy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('westerncomedy','western|comedy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('ancientcostume','costume')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('computeranimation','animation')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dramatic','drama')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('familya','family')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('familya','family')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dramedy','drama|comedy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('dramaa','drama')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('famil\|','family')



movies['GenreCorrected']=movies['GenreCorrected'].str.replace('superheroe','superhero')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('biogtaphy','biography')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('devotionalbiography','devotional|biography')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('docufiction','documentary|fiction')



movies['GenreCorrected']=movies['GenreCorrected'].str.replace('familydrama','family|drama')



movies['GenreCorrected']=movies['GenreCorrected'].str.replace('espionage','spy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('supeheroes','superhero')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('romancefiction','romance|fiction')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('horrorthriller','horror|thriller')



movies['GenreCorrected']=movies['GenreCorrected'].str.replace('suspensethriller','suspense|thriller')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('musicaliography','musical|biography')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('triller','thriller')



movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|\(fiction\)','|fiction')



movies['GenreCorrected']=movies['GenreCorrected'].str.replace('romanceaction','romance|action')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('romancecomedy','romance|comedy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('romancehorror','romance|horror')



movies['GenreCorrected']=movies['GenreCorrected'].str.replace('romcom','romance|comedy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('rom\|com','romance|comedy')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('satirical','satire')



movies['GenreCorrected']=movies['GenreCorrected'].str.replace('science_fictionchildren','science_fiction|children')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('homosexual','adult')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('sexual','adult')



movies['GenreCorrected']=movies['GenreCorrected'].str.replace('mockumentary','documentary')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('periodic','period')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('romanctic','romantic')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('politics','political')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('samurai','martial_arts')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('tv_miniseries','series')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('serial','series')



filterE = movies['GenreCorrected']=="musical–comedy"

movies.loc[filterE,'GenreCorrected'] = "musical|comedy"



filterE = movies['GenreCorrected']=="roman|porno"

movies.loc[filterE,'GenreCorrected'] = "adult"





filterE = movies['GenreCorrected']=="action—masala"

movies.loc[filterE,'GenreCorrected'] = "action|masala"





filterE = movies['GenreCorrected']=="horror–thriller"

movies.loc[filterE,'GenreCorrected'] = "horror|thriller"



movies['GenreCorrected']=movies['GenreCorrected'].str.replace('family','children')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('martial_arts','action')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('horror','thriller')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('war','action')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('adventure','action')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('science_fiction','action')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('western','action')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('western','action')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('noir','black')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('spy','action')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('superhero','action')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('social','')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('suspense','action')





filterE = movies['GenreCorrected']=="drama|romance|adult|children"

movies.loc[filterE,'GenreCorrected'] = "drama|romance|adult"



movies['GenreCorrected']=movies['GenreCorrected'].str.replace('\|–\|','|')

movies['GenreCorrected']=movies['GenreCorrected'].str.strip(to_strip='\|')

movies['GenreCorrected']=movies['GenreCorrected'].str.replace('actionner','action')

movies['GenreCorrected']=movies['GenreCorrected'].str.strip()

moviesGenre = movies[['GenreCorrected','Count']].groupby(['GenreCorrected']).count()

moviesGenre.to_csv('GenreCorrected.csv',sep=',')
movies[['GenreCorrected','Count']].groupby(['GenreCorrected'], as_index=False).count().shape[0]
movies[['GenreCorrected','Count']].groupby(['GenreCorrected'],as_index=False).count().sort_values(['Count'], ascending=False).head(10)
movies['GenreSplit']=movies['GenreCorrected'].str.split('|')

movies['GenreSplit']= movies['GenreSplit'].apply(np.sort).apply(np.unique)
movies['GenreSplit'][11]
genres_array = np.array([])



for i in range(0,movies.shape[0]-1):

    genres_array = np.concatenate((genres_array, movies['GenreSplit'][i] ))

    

genres_array
genres = pd.DataFrame({'Genre':genres_array})
genres.head(10)
# histogram for the genres is most welcome

genres['Count']=1

genres[['Genre','Count']].groupby(['Genre'], as_index=False).sum().sort_values(['Count'], ascending=False).head(10)
genres=genres[['Genre','Count']].groupby(['Genre'], as_index=False).sum().sort_values(['Count'], ascending=False)
genres = genres[genres['Genre']!='']

genres.head(25)
TotalCountGenres=sum(genres['Count'])
TotalCountGenres
genres['Frequency'] = genres['Count']/TotalCountGenres
genres['CumulativeFrequency'] = genres['Frequency'].cumsum()
genres.head(20)
np.array(genres[genres['CumulativeFrequency']<=.957]['Genre'])
genres[genres['CumulativeFrequency']<=.957][['Genre','Count']].plot(x='Genre', y='Count', kind='bar', legend=False, grid=True, figsize=(8, 5))

plt.title("Number of movies per genre")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('Movie genres', fontsize=12)

plt.show()
mainGenres=np.array(genres[genres['CumulativeFrequency']<=.957]['Genre'])
arr1=np.array(['adult', 'romance', 'drama','and'])

arr1[np.in1d(arr1,mainGenres)] # genres not in the mainGenres array will be deleted

movies['GenreSplit'][10:12].apply(lambda x: x[np.in1d(x,mainGenres)])
movies['GenreSplitMain'] = movies['GenreSplit'].apply(lambda x: x[np.in1d(x,mainGenres)])
movies[['GenreSplitMain','GenreSplit','Genre']][200:220]
# function for cleaning the plots of the movies

def clean_text(text):

    text = text.lower()

    text = re.sub(r"what's", "what is ", text)

    text = re.sub(r"\'s", " ", text)

    text = re.sub(r"\'ve", " have ", text)

    text = re.sub(r"can't", "can not ", text)

    text = re.sub(r"n't", " not ", text)

    text = re.sub(r"i'm", "i am ", text)

    text = re.sub(r"\'re", " are ", text)

    text = re.sub(r"\'d", " would ", text)

    text = re.sub(r"\'ll", " will ", text)

    text = re.sub(r"\'scuse", " excuse ", text)

    #text = re.sub('\W', ' ', text)

    #text = re.sub('\s+', ' ', text)

    text = text.strip(' ')

    return text
list(movies['Plot'][10:12].apply(clean_text))
list(movies['Plot'][10:12])
movies['PlotClean'] = movies['Plot'].apply(clean_text)
movies[['Plot','PlotClean','GenreSplitMain']][6:12]
len(movies['GenreSplitMain'][0])
movies['GenreSplitMain'][0:5].apply(len)
movies['MainGenresCount'] = movies['GenreSplitMain'].apply(len)
max(movies['MainGenresCount'] )
movies[movies['MainGenresCount']==7]
movies['MainGenresCount'].hist()



plt.title("Number of movies by number of genres")

plt.ylabel('# of movies', fontsize=12)

plt.xlabel('# of genres', fontsize=12)

plt.show()
txt = ["He is ::having a great Time, at the park time?",

       "She, unlike most women, is a big player on the park's grass.",

       "she can't be going"]

txt


# Initialize a CountVectorizer object: count_vectorizer

count_vec = CountVectorizer(stop_words="english", analyzer='word', 

                            ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None)
# Transforms the data into a bag of words

count_train = count_vec.fit(txt)

bag_of_words = count_vec.transform(txt)



# Print the first 10 features of the count_vec

print("Every feature:\n{}".format(count_vec.get_feature_names()))

print("\nEvery 3rd feature:\n{}".format(count_vec.get_feature_names()[::3]))
count_vec.fit_transform(txt).toarray()
count_vec.get_feature_names()[:3]
#z = movies.GenreSplitMain.str.split()

movies.GenreSplitMain[6:15].apply(lambda x: '-'.join(x)).str.split(pat='-',n=5,expand=True)
movies.GenreSplitMain[6:15].apply(lambda x: '-'.join(x)).str.get_dummies(sep='-')
movies.GenreSplitMain[6:15]
movies.columns
movies.shape
len(movies.Title.unique()) # the title is not unique
# The number of movies not having a genre

movies[movies.GenreCorrected==''].shape
# the dummy classes

movies = pd.concat([movies, movies.GenreSplitMain.apply(lambda x: '-'.join(x)).str.get_dummies(sep='-')], axis=1)
# the train and the test data set will be build when there is at least one genre for a movie

MoviesTrain, MoviesTest = train_test_split(movies[movies.GenreCorrected!=''], random_state=42, test_size=0.30, shuffle=True)
# definition the algorithm for feature extraction

tfidf = TfidfVectorizer(stop_words ='english', smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
# building the features

x_train = tfidf.fit_transform(MoviesTrain.PlotClean) 

x_test  = tfidf.transform(MoviesTest.PlotClean)

### for test data, the feature extraction will be done through the function transform()

### to make sure there is no features dimensionality mismatch
print('nrow of the MoviesTrain ={}'. format(MoviesTrain.shape[0]))
print('nrow of the MoviesTest ={}'. format(MoviesTest.shape[0]))
type(x_train)
x_train.toarray()
tfidf.inverse_transform(x_train[0].toarray())
print('The corpus is huge. It contain {} words.'.format(len(x_train[0].toarray()[0])))
# building the classes

y_train = MoviesTrain[MoviesTrain.columns[14:]]

y_test = MoviesTest[MoviesTest.columns[14:]]
len(y_train.columns)
len(y_test.columns)
multinomialNB=OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))
# fitting

multinomialNB.fit(x_train, y_train.action)
# compute the testing accuracy

prediction = multinomialNB.predict(x_test)
print('Test accuracy is {}'.format(accuracy_score(y_test.action, prediction)))
len(mainGenres)
accuracy_multinomialNB=pd.DataFrame(columns=['Genre', 'accuracy_multinomialNB'])

accuracy_multinomialNB.head()
i = 0

for genre in mainGenres:

    multinomialNB.fit(x_train, y_train[genre])

    prediction = multinomialNB.predict(x_test)

    accuracy_multinomialNB.loc[i,'Genre'] = genre

    accuracy_multinomialNB.loc[i,'accuracy_multinomialNB'] = accuracy_score(y_test[genre], prediction)

    i=i+1

    



    

accuracy_multinomialNB
linearSVC=OneVsRestClassifier(LinearSVC(), n_jobs=1)
accuracy_LinearSVC=pd.DataFrame(columns=['Genre', 'accuracy_LinearSVC'])

accuracy_LinearSVC.head()
  

i = 0

for genre in mainGenres:

    linearSVC.fit(x_train, y_train[genre])

    prediction = linearSVC.predict(x_test)

    accuracy_LinearSVC.loc[i,'Genre'] = genre

    accuracy_LinearSVC.loc[i,'accuracy_LinearSVC'] = accuracy_score(y_test[genre], prediction)

    i=i+1
accuracy_LinearSVC
# merging the accuracy tables

accuracy_svc_mnb = pd.merge(accuracy_multinomialNB, accuracy_LinearSVC, on='Genre', how='inner')
accuracy_svc_mnb
plt.figure(figsize=(18,10))

p1 =plt.bar(accuracy_svc_mnb.Genre, height=accuracy_svc_mnb.accuracy_multinomialNB)

p2 =plt.bar(accuracy_svc_mnb.Genre, height=accuracy_svc_mnb.accuracy_LinearSVC)

plt.xticks( rotation=90)

plt.title("Movies genre classification accuracy (multinomialNB VS LinearSVC)")

plt.ylabel('accuracy', fontsize=14)

plt.xlabel('Movie genres', fontsize=14)

plt.legend((p1[0], p2[0]), ('multinomialNB', 'LinearSVC'))

plt.show()





# the graph is not showing the comparison between the 2 classifiers.
#accuracy_multinomialNB, accuracy_LinearSVC



accuracy_multinomialNB1 = accuracy_multinomialNB

accuracy_multinomialNB1.columns = ['Genre', 'accuracy']

accuracy_multinomialNB1['classifier'] = 'multinomialNB'
accuracy_multinomialNB1
#accuracy_multinomialNB, accuracy_LinearSVC



accuracy_LinearSVC1 = accuracy_LinearSVC

accuracy_LinearSVC1.columns = ['Genre', 'accuracy']

accuracy_LinearSVC1['classifier'] = 'linearSVC'
accuracy_LinearSVC1
accu_mnb_svc = accuracy_multinomialNB1.append(accuracy_LinearSVC1)
accu_mnb_svc
sns.set(rc={'figure.figsize':(18,10)})

sns.set(style="whitegrid")

s = sns.barplot(x="Genre", y="accuracy", hue="classifier", data=accu_mnb_svc) #.set_title('Movies genre classification accuracy (multinomialNB VS LinearSVC)')

s.set_title('Movies genre classification accuracy (multinomialNB VS LinearSVC)', size=16)

s.set_xticklabels(list(mainGenres) ,rotation=45, size=15)
