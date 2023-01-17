#import cv2

import gc

import glob

import os

import json

import matplotlib.pyplot as plt

import warnings

import re



import numpy as np

import pandas as pd



#from PIL import Image



%matplotlib inline

%config InlineBackend.figure_format = 'svg'



warnings.filterwarnings("ignore")



plt.rcParams['figure.figsize'] = (12, 9)

plt.style.use('ggplot')



pd.options.display.max_rows = 64

pd.options.display.max_columns = 512
train = pd.read_csv('../input/train/train.csv')

train['AdoptionSpeed'].astype(np.int32)

test = pd.read_csv('../input/test/test.csv')

df = pd.concat([train,test],ignore_index=True)
train_sentiment_files = sorted(glob.glob('../input/train_sentiment/*.json'))

test_sentiment_files = sorted(glob.glob('../input/test_sentiment/*.json'))

sentimental_analysis = train_sentiment_files + test_sentiment_files
score=[]

magnitude=[]

petid=[]

for filename in sentimental_analysis:

    with open(filename, 'r') as f:

        sentiment_file = json.load(f)

        file_sentiment = sentiment_file['documentSentiment']

        file_score =  sentiment_file['documentSentiment']['score']

        file_magnitude = sentiment_file['documentSentiment']['magnitude']

        score.append(file_score)

        magnitude.append(file_magnitude)

        petid.append(filename.replace('.json','').replace('../input/train_sentiment/', '').replace('../input/test_sentiment/', ''))
score_dict = dict(zip(petid,score))

magnitude_dict = dict(zip(petid,magnitude))
df['Score'] = df['PetID'].map(score_dict)

df['Score'][df.Score.isnull()] = 0

df['Magnitude'] = df['PetID'].map(magnitude_dict)

df['Magnitude'][df.Magnitude.isnull()] = 0

df.set_index('PetID',inplace=True)
def namevaild(name):

    if name == np.nan:

        return 0

    elif len(str(name)) < 3:

        return 1

    elif re.match(u'[0-9]', str(name).lower()):

        return 1

    elif len(set(str(name).lower().split(' ')+['no','not','yet','male','female','unnamed'])) != len(set(str(name).lower().split(' ')))+6:

        return 1

    else:

        return 2

df['Name_state'] = df['Name'].apply(namevaild)
df['Fee_per_pet'] = df.Fee/df.Quantity



df['Fee_Bin']=pd.factorize(pd.cut(df.Fee_per_pet,bins=[0,0.01,50,100,200,500,3000],right=False))[0]

fee_bin_dummies_df = pd.get_dummies(df['Fee_Bin']).rename(columns=lambda x: 'Fee_Bin_' + str(x))

df = pd.concat([df, fee_bin_dummies_df], axis=1)
df['Quantity_Bin']=pd.factorize(pd.cut(df.Quantity,bins=[1,2,4,22],right=False))[0]

quantity_bin_dummies_df = pd.get_dummies(df['Quantity_Bin']).rename(columns=lambda x: 'Quantity_Bin_' + str(x))

df = pd.concat([df, quantity_bin_dummies_df], axis=1)
df.VideoAmt = df.VideoAmt.apply(lambda x: 1 if x > 0 else 0)



df['PhotoAmt_Bin']=pd.factorize(pd.cut(df.PhotoAmt,bins=[0,1,2,4,31],right=False))[0]

photo_bin_dummies_df = pd.get_dummies(df['PhotoAmt_Bin']).rename(columns=lambda x: 'PhotoAmt_Bin_' + str(x))

df = pd.concat([df, photo_bin_dummies_df], axis=1)
def map_state(state):

    if state == 41326:

        return 'Selangor'

    elif state == 41401:

        return 'Kuala_Lumpur'

    else:

        return 'Other_State'

df['State_Bin'] = df.State.apply(map_state)

state_bin_dummies_df = pd.get_dummies(df['State_Bin']).rename(columns=lambda x: 'State_' + str(x))

df = pd.concat([df, state_bin_dummies_df], axis=1)
rescuer_dict = df.RescuerID.value_counts().to_dict()

df['Rescuer_Num'] = df.RescuerID.map(rescuer_dict)

#df['Rescuer_Bin']=pd.factorize(pd.cut(df.Rescuer_Num,bins=[1,2,5],right=False))[0]

#df['Rescuer_Bin'].value_counts()

#rescuer_bin_dummies_df = pd.get_dummies(df['Rescuer_Bin']).rename(columns=lambda x: 'Rescuer_Bin_' + str(x))

#df = pd.concat([df, rescuer_bin_dummies_df], axis=1)
breeds = pd.read_csv('../input/breed_labels.csv')

breeds_dict = {k: v for k, v in zip(breeds['BreedID'], breeds['BreedName'])}

df['Breed1_name'] = df['Breed1'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'NA')

df['Breed2_name'] = df['Breed2'].apply(lambda x: '_'.join(breeds_dict[x].split()) if x in breeds_dict else 'NA')
df['Breed'] = df['Breed1_name'] + '--' + df['Breed2_name']

def mix_breed(string):

    breed = string.split('--')

    if breed[0] in ['Mixed_Breed','NA']:

        return 1

    elif breed[1] == 'Mixed_Breed':

        return 1

    elif breed[1] == 'NA':

        return 0

    elif breed[0] != breed[1]:

        return 1

    else:

        return 0

df['Mixed_Breed'] = df.Breed.apply(mix_breed)
'''

df.Description[df.Description.isnull()] = ''

des_list = df.Description.values.tolist()

'''
'''

import unicodedata

import re



from nltk import word_tokenize, sent_tokenize

from nltk.corpus import stopwords

from nltk.stem import LancasterStemmer, WordNetLemmatizer

from nltk.corpus import stopwords



def remove_non_ascii(words):

    """Remove non-ASCII characters from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')

        new_words.append(new_word)

    return new_words



def to_lowercase(words):

    """Convert all characters to lowercase from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = word.lower()

        new_words.append(new_word)

    return new_words



def remove_punctuation(words):

    """Remove punctuation from list of tokenized words"""

    new_words = []

    for word in words:

        new_word = re.sub(r'[^\w\s]', '', word)

        if new_word != '':

            new_words.append(new_word)

    return new_words

'''



'''

def replace_numbers(words):

    """Replace all interger occurrences in list of tokenized words with textual representation"""

    p = inflect.engine()

    new_words = []

    for word in words:

        if word.isdigit():

            new_word = p.number_to_words(word)

            new_words.append(new_word)

        else:

            new_words.append(word)

    return new_words 

'''

'''

def remove_stopwords(words):

    """Remove stop words from list of tokenized words"""

    new_words = []

    for word in words:

        if word not in stopwords.words('english'):

            new_words.append(word)

    return new_words



def stem_words(words):

    """Stem words in list of tokenized words"""

    stemmer = LancasterStemmer()

    stems = []

    for word in words:

        stem = stemmer.stem(word)

        stems.append(stem)

    return stems



def lemmatize_verbs(words):

    """Lemmatize verbs in list of tokenized words"""

    lemmatizer = WordNetLemmatizer()

    lemmas = []

    for word in words:

        lemma = lemmatizer.lemmatize(word, pos='v')

        lemmas.append(lemma)

    return lemmas



def normalize(words):

    words = remove_non_ascii(words)

    words = to_lowercase(words)

    words = remove_punctuation(words)

    #words = replace_numbers(words)

    words = remove_stopwords(words)

    #words = stem_words(words)

    words = lemmatize_verbs(words)

    return words



'''

'''

word_bag = []



for i,item in enumerate(des_list):

    words = word_tokenize(item)

    words = normalize(words)

    word_bag.append(words)

df['Word_bag'] = word_bag



def wordjoin(x):

    return ' '.join(x)



df['Word_list'] = df['Word_bag'].apply(wordjoin)

'''

'''

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer



vectorizer = CountVectorizer(min_df = 0.02)

transformer=TfidfTransformer()

tfidf = transformer.fit_transform(vectorizer.fit_transform(df.Word_list))

weight=tfidf.toarray()

'''
'''

from sklearn.decomposition import PCA



n_components = 25

pca = PCA(n_components=n_components, random_state=42)

pca.fit(weight)

text_feature = pca.transform(weight)



columns = []

for i in range(n_components):

    columns.append('text_feature_'+str(i+1))

'''
'''

df = pd.concat([df,pd.DataFrame(text_feature, index = df.index, columns = columns)],axis = 1)

'''
df.head()
df_copy = df.drop(columns=['Description','Fee','Fee_per_pet','Name','PhotoAmt','Quantity','RescuerID','State','State_Bin','Fee_Bin','Quantity_Bin','PhotoAmt_Bin','Breed','Breed1_name','Breed2_name'])

train = df_copy[df.AdoptionSpeed.notnull()]

test  = df_copy[df.AdoptionSpeed.isnull()]

print(train.shape, test.shape)
train.head()
X_train = train.drop(columns=['AdoptionSpeed'])

y_train = train.AdoptionSpeed

X_test = test.drop(columns=['AdoptionSpeed'])
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 600, max_depth=None, criterion='gini')

rf.fit(X_train,y_train)

y_predict = rf.predict(X_test).astype(np.int32)

submission = pd.DataFrame({'PetID': test.index, 'AdoptionSpeed': y_predict})

submission = submission[['PetID','AdoptionSpeed']]

submission.head()
submission.to_csv('submission.csv', index=False)