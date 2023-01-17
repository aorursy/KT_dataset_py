import nltk





#more libraries that can be will be used

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import re
#Below we read in our data for our training set and test parameters

mbti = pd.read_csv('../input/train.csv')

mbti_test= pd.read_csv('../input/test.csv')



# List of mbti types 

type_labels = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ', 

               'ISTP', 'ISFP', 'INFP', 'INTP', 

               'ESTP', 'ESFP', 'ENFP', 'ENTP', 

               'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']
mbti.head()
mbti_test.head()
plt.figure(figsize=[16, 9])

mbti['type'].value_counts().plot(kind = 'bar')

plt.show()
all_mbti = mbti
all_mbti2 = mbti_test
all_mbti=all_mbti.rename(columns={'posts':'post'})
all_mbti2=all_mbti2.rename(columns={'posts':'post'})

pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

subs_url = r'url-web'

all_mbti['post'] = all_mbti['post'].replace(to_replace = pattern_url, value = subs_url, regex = True)
all_mbti.head()
all_mbti2.head()
all_mbti2['post'] = all_mbti2['post'].replace(to_replace = pattern_url, value = subs_url, regex = True)
all_mbti2.head()
# first we make everything lower case to remove some noise from capitalisation

all_mbti['post'] = all_mbti['post'].str.lower()
all_mbti2['post'] = all_mbti2['post'].str.lower()
import string
def remove_punctuation(post):

    return ''.join([l for l in post if l not in string.punctuation])
all_mbti['post'] = all_mbti['post'].apply(remove_punctuation)
all_mbti2['post'] = all_mbti2['post'].apply(remove_punctuation)
all_mbti.head()
from nltk.tokenize import word_tokenize, TreebankWordTokenizer
# we will use the TreeBankWordTokenizer since it is MUCH quicker than the word_tokenise function

tokeniser = TreebankWordTokenizer()

all_mbti['tokens'] = all_mbti['post'].apply(tokeniser.tokenize)

all_mbti2['tokens'] = all_mbti2['post'].apply(tokeniser.tokenize)
from nltk import SnowballStemmer, PorterStemmer, LancasterStemmer
stemmer = SnowballStemmer('english')
def mbti_stemmer(words, stemmer):

    return [stemmer.stem(word) for word in words]    
# stem all words in the mbti dataframe

all_mbti['stem'] = all_mbti['tokens'].apply(mbti_stemmer, args=(stemmer, ))

all_mbti2['stem'] = all_mbti2['tokens'].apply(mbti_stemmer, args=(stemmer, ))
all_mbti['mind'] = all_mbti['type'].apply(lambda x: x[0] == 'E').astype('int')

all_mbti['energy'] = all_mbti['type'].apply(lambda x: x[1] == 'N').astype('int')

all_mbti['nature'] = all_mbti['type'].apply(lambda x: x[2] == 'T').astype('int')

all_mbti['tactics'] = all_mbti['type'].apply(lambda x: x[3] == 'J').astype('int')
all_mbti.head()
plt.figure(figsize=[16, 9])

# mind category

labels = ['Extraversion', 'Introversion']

sizes = [all_mbti['mind'].value_counts()[1], all_mbti['mind'].value_counts()[0]]



fig, ax = plt.subplots(2, 2, figsize=(8, 8))

ax[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%',

             shadow=False, startangle=90)

ax[0, 0].axis('equal')



# energy category

labels = ['Intuitive', 'Observant']

sizes = [all_mbti['energy'].value_counts()[1],all_mbti['energy'].value_counts()[0]]



ax[0, 1].pie(sizes, labels=labels, autopct='%1.1f%%',

             shadow=False, startangle=90)

ax[0, 1].axis('equal')



# nature category

labels = ['Thinking', 'Feeling']

sizes = [all_mbti['nature'].value_counts()[1], all_mbti['nature'].value_counts()[0]]



ax[1, 0].pie(sizes, labels=labels, autopct='%1.1f%%',

             shadow=False, startangle=90)

ax[1, 0].axis('equal')



# tactics category

labels = ['Judging', 'Prospecting']

sizes = [all_mbti['tactics'].value_counts()[1], all_mbti['tactics'].value_counts()[0]]



ax[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%',

             shadow=False, startangle=90)

ax[1, 1].axis('equal')

plt.tight_layout()

plt.show()
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')



lemmatizer = WordNetLemmatizer()
def mbti_lemma(words, lemmatizer):

    return [lemmatizer.lemmatize(word) for word in words]    
# lemmatize all words in dataframe

all_mbti['lemma'] = all_mbti['tokens'].apply(mbti_lemma, args=(lemmatizer, ))

all_mbti2['lemma'] = all_mbti2['tokens'].apply(mbti_lemma, args=(lemmatizer, ))
from nltk.corpus import stopwords
def bag_of_words_count(words, word_dict={}):

    """ this function takes in a list of words and returns a dictionary 

        with each word as a key, and the value represents the number of 

        times that word appeared"""

    for word in words:

        if word in word_dict.keys():

            word_dict[word] += 1

        else:

            word_dict[word] = 1

    return word_dict
# here we create a set of dictionaries

# one for each of the MBTI types

personality = {}

for pp in type_labels:

    df = all_mbti.groupby('type')

    personality[pp] = {}

    for row in df.get_group(pp)['tokens']:

        personality[pp] = bag_of_words_count(row, personality[pp])       
f=pd.DataFrame(personality)
f.head()
# next we create a list of all of the unique words...

all_words = set()

for pp in type_labels:

    for word in personality[pp]:

        all_words.add(word)
# so that we can create a dictionary of bag of words for the whole dataset

personality['all'] = {}

for pp in type_labels:    

    for word in all_words:

        if word in personality[pp].keys():

            if word in personality['all']:

                personality['all'][word] += personality[pp][word]

            else:

                personality['all'][word] = personality[pp][word]
# lets have a look at the distrbution of words

plt.hist([v for v in personality['all'].values()])
# how many words in total?

sum([v for v in personality['all'].values()])
# how many words appear only once?

len([v for v in personality['all'].values() if v == 1])
# how many words appear more than 100 times?

# how many words of the total does that account for?

print (len([v for v in personality['all'].values() if v >= 100]))

print (sum([v for v in personality['all'].values() if v >= 100]))
11013261/11681001
max_count = 100

word_index = [k for k, v in personality['all'].items() if v > max_count]
# now let's create one big data frame with the word counts by personality profile

hm = []

for p, p_bow in personality.items():

    df_bow = pd.DataFrame([(k, v) for k, v in p_bow.items() if k in word_index], columns=['Word', p])

    df_bow.set_index('Word', inplace=True)

    hm.append(df_bow)



# create one big data frame

df_bow = pd.concat(hm, axis=1)

df_bow.fillna(0, inplace=True)
# what are the top 10 words that appear most often?

df_bow.sort_values(by='all', ascending=False).head(10)
intro_types = [p for p in type_labels if p[0] == 'I']
intro_types
df_bow['I'] = df_bow[intro_types].sum(axis=1)
# convert to percentages

for col in ['I', 'all']:

    df_bow[col+'_perc'] = df_bow[col] / df_bow[col].sum()
df_bow['chi2'] = np.power((df_bow['I_perc'] - df_bow['all_perc']), 2) / df_bow['all_perc']
df_bow[['I_perc', 'all_perc', 'chi2']][df_bow['I_perc'] > df_bow['all_perc']].sort_values(by='chi2', ascending=False).head(10)
df_bow[['I_perc', 'all_perc', 'chi2']][df_bow['I_perc'] < df_bow['all_perc']].sort_values(by='chi2', ascending=False).head(20)
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(stop_words='english', ngram_range=(1, 4),max_df=0.5)
def untokenize(lst):

    return ' '.join([x for x in lst])
all_mbti['untokens']=all_mbti['tokens'].apply(untokenize)

all_mbti['unstem']=all_mbti['stem'].apply(untokenize)

all_mbti['unlemma']=all_mbti['lemma'].apply(untokenize)
column='unstem'

#column='post'

#column='untokens'

#column='unlemma'

X_count = vect.fit_transform(all_mbti[column])
from sklearn.linear_model import LogisticRegression

logreg_mind= LogisticRegression()

logreg_energy= LogisticRegression()

logreg_nature= LogisticRegression()

logreg_tactics= LogisticRegression()



X = X_count # creating our X variable
y = all_mbti['mind'] # creating our Y mind variable

y2 = all_mbti['energy'] # creating our Y energy variable

y3 = all_mbti['nature'] # creating our Y nature variable

y4 = all_mbti['tactics'] # creating our Y tactics variable
#tester

all_mbti2[column]=all_mbti2['stem'].apply(untokenize)
#tester

X_test=all_mbti2[column]
#tester

X_count_test = vect.transform(X_test)
mind_lm_log=logreg_mind.fit(X, y)
energy_lm_log= logreg_energy.fit(X, y2)

nature_lm_log= logreg_nature.fit(X, y3)

tactics_lm_log= logreg_tactics.fit(X, y4)
mind_pred= mind_lm_log.predict(X_count_test)

energy_pred= energy_lm_log.predict(X_count_test)

nature_pred=nature_lm_log.predict(X_count_test)

tactis_pred=tactics_lm_log.predict(X_count_test)
all_mbti2['mind']=pd.DataFrame(mind_pred,columns=['mind'])

all_mbti2['energy']=pd.DataFrame(energy_pred,columns=['energy'])

all_mbti2['nature']=pd.DataFrame(nature_pred,columns=['nature'])

all_mbti2['tactics']=pd.DataFrame(tactis_pred,columns=['tactics'])

submition_concept=all_mbti2[['id','mind','energy','nature','tactics']]
submition_concept.head(10)
submition_concept.to_csv('final_submit21.csv',index=False)
#from sklearn.svm import SVC



#svc_mind  = SVC(kernel='rbf')

#svc_energy = SVC(kernel='rbf')

#svc_nature = SVC(kernel='rbf')

#svc_tactics = SVC(kernel='rbf')

#mind_lm_=svc_mind.fit(X, y)

#energy_lm_= svc_energy.fit(X, y2)

#nature_lm_= svc_nature.fit(X, y3)

#tactics_lm_= svc_tactics.fit(X, y4)
#mind_pred2= mind_lm_.predict(X_count_test)

#energy_pred2= energy_lm_.predict(X_count_test)

#nature_pred2=nature_lm_.predict(X_count_test)

#tactis_pred2=tactics_lm_.predict(X_count_test)
#all_mbti2['mind2']=pd.DataFrame(mind_pred2,columns=['mind'])

#all_mbti2['energy2']=pd.DataFrame(energy_pred2,columns=['energy'])

#all_mbti2['nature2']=pd.DataFrame(nature_pred2,columns=['nature'])

#all_mbti2['tactics2']=pd.DataFrame(tactis_pred2,columns=['tactics'])


    