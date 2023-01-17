# Imports for common tasks, models imported later

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

import re

import gensim

from sklearn.metrics import fbeta_score

from imblearn.combine import SMOTETomek

import missingno as msn

from IPython.display import Image



%matplotlib inline
df = pd.read_csv("../input/cyberbully-data-formspring/formspring.csv")

# See first few rows

df.head()
# missing values visualisation

msn.bar(df);
plt.figure(figsize = (25, 8))

sns.countplot(df['userid'], palette='inferno')

plt.xticks(rotation = 90);
plt.figure(figsize = (25, 5))

sns.countplot(df['asker'][:500],

             palette = 'husl',

             hue = df['ans1'][:500]) # For convenience, see only first few rows

plt.xticks(rotation = -90);
## Generate Word Clouds

bullyDF = df[df['ans1'] == 'Yes']

Words = [str(i) for i in bullyDF['ques']]

WordsString = (" ".join(Words)).lower()

WordsString = re.sub(r'[^\w\s]', '', WordsString)



# mask

img_url = 'https://raw.githubusercontent.com/Dutta-SD/Images_Unsplash/master/cyber_bullying_images/I%20AM%20FINE.png'

from urllib.request import urlopen

from PIL import Image

## make mask

img = urlopen(img_url)

mask = np.array(Image.open(img))



# WORDCLOUD

from wordcloud import WordCloud

wc = WordCloud(width = 1200, height = 1200,

              background_color = 'white',

              mask = mask,

              contour_width=3).generate(WordsString)

plt.figure(figsize = (12, 12), facecolor = None) 

plt.imshow(wc); 

plt.axis("off");
# Posts with actual cyberbullying content

df[df['ans1'] == 'Yes'].head()
df.drop(['post', 'asker'], axis = 1, inplace = True)
# Threshold value

userid_count_threshold = 3



# Get row indexes which we will drop

drop_indexes = df[df['userid'].map(df['userid'].value_counts()) <= userid_count_threshold].index



df_dropped = df.drop(drop_indexes)
def impute_ans_columns(value):

    if value == 'Yes':

        return 1

    return 0



for col in ['ans1', 'ans2', 'ans3']:

    df_dropped[col] = df[col].apply(impute_ans_columns)

df_dropped.head()
def impute_severity_columns(value):

    '''Value will be a string. We need to convert it to int'''

    try:

        return int(value)

    except ValueError as e:

        return 0



for col in ['severity1', 'severity2', 'severity3']:

    df_dropped[col] = df_dropped[col].apply(impute_severity_columns)
df_dropped.tail()
df_dropped['IsBully'] = (

    (df_dropped.ans1 * df_dropped.severity1 + df_dropped.ans2 * df_dropped.severity2 + df_dropped.ans3 * df_dropped.severity3) / 30) >= 0.2



# Remove uneccessary columns

df_2 = df_dropped.drop(['ans1', 'severity1','bully1','ans2','severity2','bully2','ans3','severity3','bully3'], axis = 1)
df_2.head()
for col in ['ques', 'ans']:

    df_2[col] = df_2[col].str.replace("&#039;", "'") # Put back the apostrophe



    df_2[col] = df_2[col].str.replace("<br>", "") 

    df_2[col] = df_2[col].str.replace("&quot;", "") 

    #df_2[col] = df_2[col].str.replace("<3", "love")

    

df_2.head()
from sklearn.model_selection import train_test_split

X, y = df_2.iloc[:, :-1], df_2.iloc[:, -1]



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.25, random_state=0, shuffle=True)



# reset indices as indices hold no value for us

X_train = X_train.reset_index(drop = True)

X_val = X_val.reset_index(drop = True)

y_train = y_train.reset_index(drop = True)

y_val = y_val.reset_index(drop = True)
from collections import Counter

counts_userid = dict(Counter(X_train['userid']))



for key in counts_userid.keys():

    # Log transform

    counts_userid[key] = np.log10(counts_userid[key] / len(X_train))



X_train['userid'] = X_train['userid'].map(counts_userid)

X_val['userid'] = X_val['userid'].map(counts_userid)
def tokenize(text):

    stop_words = stopwords.words("english")

    lemmatizer = WordNetLemmatizer()

    

    # normalize case and remove punctuation

    text = re.sub(r"[^a-zA-Z0-9]", " ", str(text).lower())

    

    # tokenize text

    tokens = word_tokenize(text)

    

    # lemmatize andremove stop words

    tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens if word not in stop_words]



    return tokens
## getting GloVe word2vectors



from gensim.scripts.glove2word2vec import glove2word2vec

# convert txt to word2vec format for easy access

glove_input_file = '../input/glove-global-vectors-for-word-representation/glove.6B.50d.txt'

word2vec_output_file = 'glove.6B.50d.txt.word2vec'

glove2word2vec(glove_input_file, word2vec_output_file)



from gensim.models import KeyedVectors

# load the Stanford GloVe model

filename = './glove.6B.50d.txt.word2vec'

model = KeyedVectors.load_word2vec_format(filename, binary=False)
def putWordVector(text):

    '''Returns Word Vectors for passed unclean string'''

    clean_text = tokenize(text) # list of words

    wordvecFinal = np.zeros((50,), dtype=np.float32)

    

    for word in clean_text:

        try:

            word_vec = model[word]

            wordvecFinal = np.add(word_vec, wordvecFinal)

        except KeyError as e:

            continue

    return wordvecFinal
def addWordVectors(df, colName):

    ''' Adds word vectors to the dframe,

    returns a dataframe.

    values - pandas Series

    colName - Name of the column which contains string'''

    df_new = df[colName].apply(putWordVector)

    

    columnNames = [colName + str(i) for i in range(50)]



    df_new = pd.DataFrame(df_new.values.tolist(), columns = columnNames )

    df_new = pd.concat([df, df_new], axis = 1)

    df_new = df_new.drop([colName], axis = 1)

    return df_new 
X_train, X_val = addWordVectors(addWordVectors(X_train, 'ques'), 'ans'), addWordVectors(addWordVectors(X_val, 'ques'), 'ans')

X_train.head()
smk = SMOTETomek()

X_train, y_train = smk.fit_sample(X_train.values, y_train.values)
from sklearn.preprocessing import StandardScaler

from sklearn.feature_selection import SelectKBest



skbest = SelectKBest()

sc = StandardScaler()



X_train = skbest.fit_transform(sc.fit_transform(X_train), y_train)

X_val = skbest.transform(sc.transform(X_val))
from sklearn.metrics import plot_roc_curve

from xgboost import XGBClassifier

from sklearn.ensemble import RandomForestClassifier, StackingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression, SGDClassifier



estimators = [

    ('xgboost_model', XGBClassifier()),

    ('randomForest_model', RandomForestClassifier()),

    ('naive_bayes', GaussianNB()),

    ('SGD', SGDClassifier())

]



finalEstimator = LogisticRegression()
model = StackingClassifier(estimators=estimators,

                           final_estimator = finalEstimator,

                           cv = 5,

                           n_jobs = -1)
predictions = model.fit(X_train, y_train).predict(X_val)
plot_roc_curve(model, X_val, y_val);