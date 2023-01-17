import pandas as pd

import numpy as np

from collections import defaultdict

#import lightgbm as lgb

from scipy.sparse import vstack, hstack, csr_matrix, spmatrix

from scipy.stats import binom

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import CountVectorizer as CV

import datetime

import gc

import re



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score,roc_auc_score,accuracy_score

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

#from nltk.stem.porter import PorterStemmer

#from nltk.corpus import stopwords

#from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from tqdm import tqdm

from gensim.models import Word2Vec

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import TimeSeriesSplit

import math

from nltk.stem import PorterStemmer

from nltk.stem.snowball import SnowballStemmer

from nltk.stem.wordnet import WordNetLemmatizer

import xgboost as xgb

import seaborn as sns

from sklearn.multioutput import MultiOutputRegressor



from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import pickle



from tqdm import tqdm

import os

from nltk.corpus import stopwords



import gensim

from gensim.utils import simple_preprocess

from gensim import corpora

from gensim.models import LdaModel



from sklearn.decomposition import NMF, LatentDirichletAllocation



import warnings

warnings.filterwarnings("ignore")



import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier

from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics.classification import accuracy_score, log_loss



from nltk.sentiment.vader import SentimentIntensityAnalyzer

from tqdm import tqdm

import nltk

from sklearn.model_selection import train_test_split



from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential

from keras.layers import Dense, Dropout, BatchNormalization, Flatten, concatenate, GlobalMaxPooling1D, add 

from keras.layers import LSTM, SpatialDropout1D, Input, Dense, Bidirectional, CuDNNLSTM, GlobalAveragePooling1D

from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor

from keras.utils import np_utils

from scipy.sparse import coo_matrix

from sklearn.preprocessing import StandardScaler



from keras.models import Model

from keras.models import Sequential

from keras.layers.embeddings import Embedding

from keras.preprocessing import sequence

from keras.preprocessing.text import Tokenizer

for dirname, _, filenames in os.walk('/kaggle/output'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_columns', None)
data = pd.read_csv('/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
# denoting duplicate rows

data[data.duplicated(['comment_text','target'], keep=False)]
data.shape
data[data['id']==240344]
data[data['id']==282368]
data.head()
data.describe()
def convert_to_bool(df, col_name, col_bool):

    df[col_bool] = np.where(df[col_name] >= 0.5, 1, 0)       



def convert_dataframe_to_bool(df, columns, col_bool):        

    bool_df = df.copy()

    convert_to_bool(bool_df, columns, col_bool)

    #for col in columns:

        #convert_to_bool(bool_df, col)

    return bool_df

data.fillna(0, inplace = True)



data = convert_dataframe_to_bool(data, ['target'], 'target_bool')

data.head()
print("\nTotal number of points in both classes:")



data['target_bool'].value_counts()
fig = plt.figure(figsize=(10,10))

data.hist(column='target')

plt.xlabel("Target/Toxicity level")

plt.ylabel("No of comments")

plt.show()
sorted_data=data.sort_values('publication_id', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
final=sorted_data.drop_duplicates(subset={"comment_text","target_bool"}, keep='first', inplace=False)

final.shape
#Checking to see how much % of data still remains



(final['id'].size*1.0)/(data['id'].size*1.0)*100
final['target_bool'].value_counts()
contraction_mapping = {

    "Trump's" : 'trump is',"'cause": 'because',',cause': 'because',';cause': 'because',"ain't": 'am not','ain,t': 'am not',

    'ain;t': 'am not','ain´t': 'am not','ain’t': 'am not',"aren't": 'are not',

    'aren,t': 'are not','aren;t': 'are not','aren´t': 'are not','aren’t': 'are not',"can't": 'cannot',"can't've": 'cannot have','can,t': 'cannot','can,t,ve': 'cannot have',

    'can;t': 'cannot','can;t;ve': 'cannot have',

    'can´t': 'cannot','can´t´ve': 'cannot have','can’t': 'cannot','can’t’ve': 'cannot have',

    "could've": 'could have','could,ve': 'could have','could;ve': 'could have',"couldn't": 'could not',"couldn't've": 'could not have','couldn,t': 'could not','couldn,t,ve': 'could not have','couldn;t': 'could not',

    'couldn;t;ve': 'could not have','couldn´t': 'could not',

    'couldn´t´ve': 'could not have','couldn’t': 'could not','couldn’t’ve': 'could not have','could´ve': 'could have',

    'could’ve': 'could have',"didn't": 'did not','didn,t': 'did not','didn;t': 'did not','didn´t': 'did not',

    'didn’t': 'did not',"doesn't": 'does not','doesn,t': 'does not','doesn;t': 'does not','doesn´t': 'does not',

    'doesn’t': 'does not',"don't": 'do not',"Don't": 'do not','don,t': 'do not','don;t': 'do not','don´t': 'do not',"They're":'they are','don’t': 'do not','Don’t': 'do not',

    "hadn't": 'had not',"hadn't've": 'had not have','hadn,t': 'had not','hadn,t,ve': 'had not have','hadn;t': 'had not',

    'hadn;t;ve': 'had not have','hadn´t': 'had not','hadn´t´ve': 'had not have','hadn’t': 'had not','hadn’t’ve': 'had not have',"hasn't": 'has not','hasn,t': 'has not','hasn;t': 'has not','hasn´t': 'has not','hasn’t': 'has not',

    "haven't": 'have not','haven,t': 'have not','haven;t': 'have not','haven´t': 'have not','haven’t': 'have not',"he'd": 'he would',

    "he'd've": 'he would have',"he'll": 'he will',

    "he's": 'he is',"He's": 'he is','he,d': 'he would','he,d,ve': 'he would have','he,ll': 'he will','he,s': 'he is','he;d': 'he would',

    'he;d;ve': 'he would have','he;ll': 'he will','he;s': 'he is','he´d': 'he would','he´d´ve': 'he would have','he´ll': 'he will',

    'he´s': 'he is','he’d': 'he would','he’d’ve': 'he would have','he’ll': 'he will','he’s': 'he is',"how'd": 'how did',"how'll": 'how will',

    "how's": 'how is','how,d': 'how did','how,ll': 'how will','how,s': 'how is','how;d': 'how did','how;ll': 'how will',

    'how;s': 'how is','how´d': 'how did','how´ll': 'how will','how´s': 'how is','how’d': 'how did','how’ll': 'how will',

    'how’s': 'how is',"i'd": 'i would',"I'd": 'i would',"i'll": 'i will',"I'll": 'i will',"i'm": 'i am',"I'm": 'i am',"i've": 'i have',"I've": 'i have','i,d': 'i would','i,ll': 'i will',

    'i,m': 'i am','i,ve': 'i have','i;d': 'i would','i;ll': 'i will','i;m': 'i am','i;ve': 'i have',"isn't": 'is not',

    'isn,t': 'is not','isn;t': 'is not','isn´t': 'is not','isn’t': 'is not',"it'd": 'it would',"it'll": 'it will',"It's":'it is',

    "it's": 'it is','it,d': 'it would','it,ll': 'it will','it,s': 'it is','it;d': 'it would','it;ll': 'it will','it;s': 'it is','it´d': 'it would','it´ll': 'it will','it´s': 'it is',

    'it’d': 'it would','it’ll': 'it will','it’s': 'it is',"It's":'it is',

    'i´d': 'i would','i´ll': 'i will','i´m': 'i am','i´ve': 'i have','i’d': 'i would','i’ll': 'i will','i’m': 'i am','I’m': 'i am',

    'i’ve': 'i have','I’ve': 'i have',"let's": 'let us','let,s': 'let us','let;s': 'let us','let´s': 'let us',

    'let’s': 'let us',"ma'am": 'madam','ma,am': 'madam','ma;am': 'madam',"mayn't": 'may not','mayn,t': 'may not','mayn;t': 'may not',

    'mayn´t': 'may not','mayn’t': 'may not','ma´am': 'madam','ma’am': 'madam',"might've": 'might have','might,ve': 'might have','might;ve': 'might have',"mightn't": 'might not','mightn,t': 'might not','mightn;t': 'might not','mightn´t': 'might not',

    'mightn’t': 'might not','might´ve': 'might have','might’ve': 'might have',"must've": 'must have','must,ve': 'must have','must;ve': 'must have',

    "mustn't": 'must not','mustn,t': 'must not','mustn;t': 'must not','mustn´t': 'must not','mustn’t': 'must not','must´ve': 'must have',

    'must’ve': 'must have',"needn't": 'need not','needn,t': 'need not','needn;t': 'need not','needn´t': 'need not','needn’t': 'need not',"oughtn't": 'ought not','oughtn,t': 'ought not','oughtn;t': 'ought not',

    'oughtn´t': 'ought not','oughtn’t': 'ought not',"sha'n't": 'shall not','sha,n,t': 'shall not','sha;n;t': 'shall not',"shan't": 'shall not',

    'shan,t': 'shall not','shan;t': 'shall not','shan´t': 'shall not','shan’t': 'shall not','sha´n´t': 'shall not','sha’n’t': 'shall not',

    "she'd": 'she would',"she'll": 'she will',"she's": 'she is','she,d': 'she would','she,ll': 'she will',

    'she,s': 'she is','she;d': 'she would','she;ll': 'she will','she;s': 'she is','she´d': 'she would','she´ll': 'she will',

    'she´s': 'she is','she’d': 'she would','she’ll': 'she will','she’s': 'she is',"should've": 'should have','should,ve': 'should have','should;ve': 'should have',

    "shouldn't": 'should not','shouldn,t': 'should not','shouldn;t': 'should not','shouldn´t': 'should not','shouldn’t': 'should not','should´ve': 'should have',

    'should’ve': 'should have',"that'd": 'that would',"that's": 'that is','that,d': 'that would','that,s': 'that is','that;d': 'that would',

    'that;s': 'that is','that´d': 'that would','that´s': 'that is','that’d': 'that would','that’s': 'that is',"there'd": 'there had',

    "there's": 'there is','there,d': 'there had','there,s': 'there is','there;d': 'there had','there;s': 'there is',

    'there´d': 'there had','there´s': 'there is','there’d': 'there had','there’s': 'there is',

    "they'd": 'they would',"they'll": 'they will',"they're": 'they are',"they've": 'they have',

    'they,d': 'they would','they,ll': 'they will','they,re': 'they are','they,ve': 'they have','they;d': 'they would','they;ll': 'they will','they;re': 'they are',

    'they;ve': 'they have','they´d': 'they would','they´ll': 'they will','they´re': 'they are','they´ve': 'they have','they’d': 'they would','they’ll': 'they will',

    'they’re': 'they are','they’ve': 'they have',"wasn't": 'was not','wasn,t': 'was not','wasn;t': 'was not','wasn´t': 'was not',

    'wasn’t': 'was not',"we'd": 'we would',"we'll": 'we will',"we're": 'we are',"we've": 'we have',"We've": 'we have','we,d': 'we would','we,ll': 'we will',

    'we,re': 'we are','we,ve': 'we have','we;d': 'we would','we;ll': 'we will','we;re': 'we are','we;ve': 'we have',

    "weren't": 'were not','weren,t': 'were not','weren;t': 'were not','weren´t': 'were not','weren’t': 'were not','we´d': 'we would','we´ll': 'we will',

    'we´re': 'we are','we´ve': 'we have','we’d': 'we would','we’ll': 'we will','we’re': 'we are','we’ve': 'we have','We’ve': 'we have',"what'll": 'what will',"what're": 'what are',"what's": 'what is',

    "what've": 'what have','what,ll': 'what will','what,re': 'what are','what,s': 'what is','what,ve': 'what have','what;ll': 'what will','what;re': 'what are',

    'what;s': 'what is','what;ve': 'what have','what´ll': 'what will',

    'what´re': 'what are','what´s': 'what is','what´ve': 'what have','what’ll': 'what will','what’re': 'what are','what’s': 'what is',

    'what’ve': 'what have',"where'd": 'where did',"where's": 'where is','where,d': 'where did','where,s': 'where is','where;d': 'where did',

    'where;s': 'where is','where´d': 'where did','where´s': 'where is','where’d': 'where did','where’s': 'where is',

    "who'll": 'who will',"who's": 'who is','who,ll': 'who will','who,s': 'who is','who;ll': 'who will','who;s': 'who is',

    'who´ll': 'who will','who´s': 'who is','who’ll': 'who will','who’s': 'who is',"won't": 'will not','won,t': 'will not','won;t': 'will not',

    'won´t': 'will not','won’t': 'will not',"wouldn't": 'would not','wouldn,t': 'would not','wouldn;t': 'would not','wouldn´t': 'would not',

    'wouldn’t': 'would not',"you'd": 'you would',"you'll": 'you will',"you're": 'you are','you,d': 'you would','you,ll': 'you will',

    'you,re': 'you are','you;d': 'you would','you;ll': 'you will',

    'you;re': 'you are','you´d': 'you would','you´ll': 'you will','you´re': 'you are','you’d': 'you would','you’ll': 'you will','you’re': 'you are',

    '´cause': 'because','’cause': 'because',"you've": "you have","could'nt": 'could not',

    "havn't": 'have not',"here’s": "here is",'i""m': 'i am',"i'am": 'i am',"i'l": "i will","i'v": 'i have',"wan't": 'want',"was'nt": "was not","who'd": "who would",

    "who're": "who are","who've": "who have","why'd": "why would","would've": "would have","y'all": "you all","y'know": "you know","you.i": "you i",

    "your'e": "you are","arn't": "are not","agains't": "against","c'mon": "common","doens't": "does not",'don""t': "do not","dosen't": "does not",

    "dosn't": "does not","shoudn't": "should not","that'll": "that will","there'll": "there will","there're": "there are",

    "this'll": "this all","u're": "you are", "ya'll": "you all","you'r": "you are","you’ve": "you have","d'int": "did not","did'nt": "did not","din't": "did not","dont't": "do not","gov't": "government",

    "i'ma": "i am","is'nt": "is not","‘I":'I',

    'ᴀɴᴅ':'and','ᴛʜᴇ':'the','ʜᴏᴍᴇ':'home','ᴜᴘ':'up','ʙʏ':'by','ᴀᴛ':'at','…and':'and','civilbeat':'civil beat',\

    'TrumpCare':'Trump care','Trumpcare':'Trump care', 'OBAMAcare':'Obama care','ᴄʜᴇᴄᴋ':'check','ғᴏʀ':'for','ᴛʜɪs':'this','ᴄᴏᴍᴘᴜᴛᴇʀ':'computer',\

    'ᴍᴏɴᴛʜ':'month','ᴡᴏʀᴋɪɴɢ':'working','ᴊᴏʙ':'job','ғʀᴏᴍ':'from','Sᴛᴀʀᴛ':'start','gubmit':'submit','CO₂':'carbon dioxide','ғɪʀsᴛ':'first',\

    'ᴇɴᴅ':'end','ᴄᴀɴ':'can','ʜᴀᴠᴇ':'have','ᴛᴏ':'to','ʟɪɴᴋ':'link','ᴏғ':'of','ʜᴏᴜʀʟʏ':'hourly','ᴡᴇᴇᴋ':'week','ᴇɴᴅ':'end','ᴇxᴛʀᴀ':'extra',\

    'Gʀᴇᴀᴛ':'great','sᴛᴜᴅᴇɴᴛs':'student','sᴛᴀʏ':'stay','ᴍᴏᴍs':'mother','ᴏʀ':'or','ᴀɴʏᴏɴᴇ':'anyone','ɴᴇᴇᴅɪɴɢ':'needing','ᴀɴ':'an','ɪɴᴄᴏᴍᴇ':'income',\

    'ʀᴇʟɪᴀʙʟᴇ':'reliable','ғɪʀsᴛ':'first','ʏᴏᴜʀ':'your','sɪɢɴɪɴɢ':'signing','ʙᴏᴛᴛᴏᴍ':'bottom','ғᴏʟʟᴏᴡɪɴɢ':'following','Mᴀᴋᴇ':'make',\

    'ᴄᴏɴɴᴇᴄᴛɪᴏɴ':'connection','ɪɴᴛᴇʀɴᴇᴛ':'internet','financialpost':'financial post', 'ʜaᴠᴇ':' have ', 'ᴄaɴ':' can ', 'Maᴋᴇ':' make ', 'ʀᴇʟɪaʙʟᴇ':' reliable ', 'ɴᴇᴇᴅ':' need ',

    'ᴏɴʟʏ':' only ', 'ᴇxᴛʀa':' extra ', 'aɴ':' an ', 'aɴʏᴏɴᴇ':' anyone ', 'sᴛaʏ':' stay ', 'Sᴛaʀᴛ':' start', 'SHOPO':'shop'," :‑)":'smiley',\

    " :)":'smiley'," :-]":'smiley'," :]":'smiley'," :‑D":'laughing'," :D":'laughing'," =D":'laughing'," :‑(":'sad'," :(":'sad'," ;)":'wink'," :P":'cheeky'

    }
ps = PorterStemmer()

ps.stem('inducing')
sn = SnowballStemmer("english")

sn.stem('inducing')
lm = WordNetLemmatizer()

ps.stem(lm.lemmatize('studies'))
"".join([lm.lemmatize(token) for token in "Thank you!! This would make my life a lot less anxiety inducing. Keep it up, and don't let anyone get in your way!"])
stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\

            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \

            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\

            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \

            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \

            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \

            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\

            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\

            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\

            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \

            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \

            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\

            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\

            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \

            'won', "won't", 'wouldn', "wouldn't"])
ps = PorterStemmer() 

lemmatizer = WordNetLemmatizer()

sn = SnowballStemmer("english")





def clean_contractions(text, mapping):

    text = re.sub(r"http\S+", "", text)

    text = re.sub("\S*\d\S*", "", text).strip()

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'

    for s in punct:

        text = text.replace(s, " ")

    text = text.replace('\n', " ")

    text = re.sub('[^A-Za-z]+', ' ', text)

    #for t in text.split():

        #text = text.replace(t, lemmatizer.lemmatize(t))

        #text = text.replace(t, sn.stem(t))

    #text = ' '.join(e.lower() for e in text.split() if e.lower() not in stopwords)

    text = ' '.join(e.lower() for e in text.split())

    return text



print("cleaning started at ",datetime.datetime.now())

final['treated_comment'] = final['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))

print("cleaning ended at ",datetime.datetime.now())
final['treated_comment'] = final['treated_comment'].astype('category')
final['comment_text'] = final['comment_text'].astype('category')
final.head()
final['treated_comment'].iloc[1]
final['comment_text'].iloc[1]
col_list = ['treated_comment', 'comment_text', 'target_bool', 'target']
num=final.shape[0]

final=final.sort_values(by=['created_date'], ascending=True)

train=final[col_list].iloc[:math.ceil(num*0.8),:]

test=final[col_list].iloc[math.ceil(num*0.8):,:]
train.head()
train['target_bool'].value_counts()
test['target_bool'].value_counts()
df_train, cv_df, y_train, y_cv = train_test_split(train, train['target_bool'],

                                                  stratify=train['target_bool'], test_size=0.20, random_state=42)
y_train.value_counts()
df_test=test

y_test=test['target_bool']
print(df_train.shape)

print(df_test.shape)

print(y_train.shape)

print(y_test.shape)

print(cv_df.shape)

print(y_cv.shape)
print('Number of data points in train data:', df_train.shape[0])

print('Number of data points in test data:', df_test.shape[0])

print('Number of data points of Y label in train data:', y_train.shape[0])

print('Number of data points of Y label in test data:', y_test.shape[0])
max_words = 100000

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(df_train['treated_comment'])





sequences_text_train = tokenizer.texts_to_sequences(df_train['treated_comment'])

sequences_text_test = tokenizer.texts_to_sequences(df_test['treated_comment'])

sequences_text_cv = tokenizer.texts_to_sequences(cv_df['treated_comment'])



max_len = max(len(l) for l in sequences_text_train)

pad_train = sequence.pad_sequences(sequences_text_train, maxlen=max_len)

pad_test = sequence.pad_sequences(sequences_text_test, maxlen=max_len)

pad_cv = sequence.pad_sequences(sequences_text_cv, maxlen=max_len)
words = Input(shape=(None,))

x = Embedding(max_words, 128, input_length=max_len)(words)

x = SpatialDropout1D(0.2)(x)

x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)

x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)



hidden = concatenate([

     GlobalMaxPooling1D()(x),

     GlobalAveragePooling1D()(x),

])

hidden = add([hidden, Dense(4 * 128, activation='relu')(hidden)])

hidden = add([hidden, Dense(4 * 128, activation='relu')(hidden)])

result = Dense(1, activation='sigmoid')(hidden)



model = Model(inputs=words, outputs=[result])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae','acc'])
'''

model = Sequential()

model.add(Embedding(max_words, 128, input_length=max_len))

model.add(Bidirectional(CuDNNLSTM(128, return_sequences=True)))

model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))

model.add(Dropout(0.5))

model.add(BatchNormalization())

model.add(Flatten())

model.add(Dense(32, activation='relu'))

model.add(BatchNormalization())

model.add(Dropout(0.3))

model.add(Dense(16, activation='relu'))

model.add(BatchNormalization())

model.add(Dense(1, activation='sigmoid'))



# model compile

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae','acc'])

model.summary()

'''
print(datetime.datetime.now())



df_train=''

df_test=''

cv_df=''



history = model.fit(pad_train, y_train.values,verbose=2, epochs=2, batch_size=2048,

                    validation_data=(pad_cv, y_cv.values))

print(datetime.datetime.now())
y_pred = model.predict(pad_test)

y_pred=np.where(np.asarray(y_pred) >= 0.5, 1, 0)



print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))

print("\tRecall: %1.3f" % recall_score(y_test, y_pred))

print("\tF1: %1.3f" % f1_score(y_test, y_pred))

print("\tAccuracy: %1.3f" % accuracy_score(y_test, y_pred))



y_prob = model.predict(pad_test)

print("y_prob: ",y_prob)

print("\tROC_AUC: %1.3f" % roc_auc_score(y_test, y_prob))





print(datetime.datetime.now())

df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(2),range(2))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g');
test_data = pd.read_csv('/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/test.csv')

test_data_cleaned = test_data['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))

sequences_text_test_data = tokenizer.texts_to_sequences(test_data_cleaned)

pad_test_data = sequence.pad_sequences(sequences_text_test_data, maxlen=max_len)

test_data_pred=model.predict(pad_test_data)
df_submit = pd.read_csv('/kaggle/input/jigsaw-unintended-bias-in-toxicity-classification/sample_submission.csv')

df_submit.prediction = test_data_pred

df_submit.to_csv('submission.csv', index=False)