#Standard Python libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#Natural language processing libraries

import nltk

import re



import time



#Interactive computing

from IPython.core.magics.execution import _format_time

from IPython.display import display as d

from IPython.display import Audio

from IPython.core.display import HTML



#Accountability

import logging as log



import os

print(os.listdir("../input"))
train_df = pd.read_csv('../input/train.csv')

train_df.head()
test = pd.read_csv('../input/test.csv')

test.head()
#Checking if we have all sixteen personality types represented

Personalities = train_df.groupby("type").count()

Personalities.sort_values("posts", ascending=False, inplace=True)

Personalities.index.values
#Visualizing the distribution of the personality types

count_types = train_df['type'].value_counts()

plt.figure(figsize=(10,5))

sns.barplot(count_types.index, count_types.values, alpha=1, palette="winter")

plt.ylabel('No of persons', fontsize=12)

plt.xlabel('personality Types', fontsize=12)

plt.title('Distribution of personality types')

plt.show()
def alert():

    """ makes sound on client using javascript"""  

    

    framerate = 44100

    duration=0.5

    freq=340

    t = np.linspace(0,duration,framerate*duration)

    data = np.sin(2*np.pi*freq*t)

    d(Audio(data,rate=framerate, autoplay=True))
def link_transformer(df, column, reports=True):

    """Search over a column in a pandas dataframe for urls.

    

    extract the title related to the url then replace the url with the title.

    

    

    df : pandas Dataframe object

    

    column: string type object equal to the exact name of the colum you want to replace the urls

    

    reports: Boolean Value (default=True)

        If true give active report updates on the last index completed and the ammount of reported fail title extractions

   

    """

    

    total_errors = 0

    count = 0

    from mechanize import Browser

    br = Browser()

    

    while count != len(df):

        errors = 0

        

        url = re.findall(r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+', df.loc[count, column])

        

        for link in url:

            try: 

                br.open(link)

                df.loc[count, column] = df.loc[count, column].replace(link, br.title())

            except:

                

                if reports == True:

                    print(f'failed--- {link}')

                elif reports == False:

                    pass

                else:

                    raise ValueError('reports expected a boolean value')

                

                total_errors += 1

                errors += 1

                

                continue

                

        if reports == True:

            if errors == 0:

                report = 'no errors'

                errors = ''

            elif errors == 1:

                report = 'error'

            else:

                report = 'errors'

            print(f'\nIndex {count + 1} completed. {errors} {report} reported\n______________________\n\n')

    

        elif reports == False:

            pass

        

        else:

            raise ValueError('reports expected a boolean value')

                

        

        count += 1

    print(f'{total_errors} total errors throughout full runtime')

    

#example



#sample = pd.read_csv('train.csv').sample(3, random_state=20).reset_index(drop=True)

#sample



#link_transformer(sample, 'posts')



#sample
def remove_links(df, column):

    """Replace urls by searching for the characters normally found in urls 

    and replace the string found with the string web-link

    

    df : pandas Dataframe object

    

    column: string type object equal to the exact name of the colum you want to replace the urls

    """

    

    return df[column].replace(r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+', 

                           r'web-link', regex=True, inplace=True)
def no_punc_num(post):

    """The function imports punctuation and define numbers then removes them from a column in a dataframe 

    using a for loop.

    

    to use, use pd.DataFrame.apply() on a Dataframe object"""

    

    from string import punctuation

    pun_num = punctuation + '1234567890'

    return ''.join([letter for letter in post if letter not in pun_num])
tokens = nltk.tokenize.TreebankWordTokenizer()
lem = nltk.stem.WordNetLemmatizer()

def lemmatized(words, lemmatizer):

    """Transform a list of words into base forms 

    

    example: hippopotami = hippopotamus





    Required imports  

   ------------------

    nltk.stem.WordNetLemmatizer()

    

    

    Parameters 

   ------------

   lemmatizer: nltk.stem.WordNetLemmatizer() object

   

   

   to use, use pd.DataFrame.apply() on a Dataframe object 

    """

    return [lemmatizer.lemmatize(word) for word in words]
def remove_stop_words(tokens):

    """Removes a list of words from a Dataframe

    

    Required imports  

   ------------------

    A list of stopwords to remove

    

    

    to use, use pd.DataFrame.apply() on a Dataframe object  

    """

    return [t for t in tokens if t not in stopwords]
#Splitting sentences

train = []

for types, posts in train_df.iterrows():

    for split_at in posts['posts'].split('|||'):

        train.append([posts['type'], split_at])

train = pd.DataFrame(train, columns=['type', 'post'])
train.head()
#making all the words lowwer case

train.post = train.post.str.lower()
#removing punctuation and numbers

train.post = train.post.apply(no_punc_num)
#tokenizing words

train['tokenized'] = train.post.apply(tokens.tokenize)
#lemmatizing words

train['lemmatized'] = train.tokenized.apply(lemmatized, args=(lem,))
def bag_count(word, bag={}):

    '''text vectorize by representing every word as a integer and counting the frequency of appearence'''

    for w in word:

        if w not in bag.keys():

            bag[w] = 1

        else:

            bag[w] += 1

    return bag



per_type = {}

for pt in list(train.type.unique()):

    df = train.groupby('type')

    per_type[pt] = {}

    for row in df.get_group(pt)['lemmatized']:

        per_type[pt] = bag_count(row, per_type[pt])



len(per_type.keys())
#creating a list of unique words

unique_words = set()

for pt in list(train.type.unique()):

    for word in per_type[pt]:

        unique_words.add(word)
unique_words
personality_stop_words = list(per_type.keys())
#finding the frequency of words

per_type['all'] = {}

for tp in list(train.type.unique()):

    for word in unique_words:

        if word in per_type[tp].keys():

            if word not in per_type['all']:

                per_type['all'][word] = per_type[tp][word]

            else:

                per_type['all'][word] += per_type[tp][word] 
per_type['all']
print(len(per_type['all']))
#Appearence of a word longer that 2 standard deviations in percentage

(sum([v for v in per_type['all'].values() if v >= 43]))/sum([v for v in per_type['all'].values()])
#Checking the words

word_index = [k for k, v in per_type['all'].items() if v > 43]
#using for loop to find word usage per type

per_type_words = []

for pt, p_word in per_type.items():

    word_useage = pd.DataFrame([(k, v) for k, v in p_word.items() if k in word_index], columns=['Word', pt])

    word_useage.set_index('Word', inplace=True)

    per_type_words.append(word_useage)
word_useage = pd.concat(per_type_words, axis=1)

word_useage.fillna(0, inplace=True)
word_useage.sample(10)
personality_stop_words
#Finding sum of the word usage and identifying them to each variable



I = [x for x in personality_stop_words if x[0] == 'I']

E = [x for x in personality_stop_words if x[0] == 'E']

word_useage['I'] = word_useage[I].sum(axis=1)

word_useage['E'] = word_useage[E].sum(axis=1)



S = [x for x in personality_stop_words if x[1] == 'S']

N = [x for x in personality_stop_words if x[1] == 'N']

word_useage['S'] = word_useage[S].sum(axis=1)

word_useage['N'] = word_useage[N].sum(axis=1)



F = [x for x in personality_stop_words if x[2] == 'F']

T = [x for x in personality_stop_words if x[2] == 'T']

word_useage['F'] = word_useage[F].sum(axis=1)

word_useage['T'] = word_useage[T].sum(axis=1)



P = [x for x in personality_stop_words if x[3] == 'P']

J = [x for x in personality_stop_words if x[3] == 'J']

word_useage['P'] = word_useage[P].sum(axis=1)

word_useage['J'] = word_useage[J].sum(axis=1)
word_useage.sample(10)
#Word usage in percentage form

for col in ['I', 'all']:

    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()

for col in ['E', 'all']:

    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()



for col in ['S', 'all']:

    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()

for col in ['N', 'all']:

    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()



for col in ['F', 'all']:

    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()

for col in ['T', 'all']:

    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()



for col in ['P', 'all']:

    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()

for col in ['J', 'all']:

    word_useage[col+'_perc'] = word_useage[col] / word_useage[col].sum()
word_useage.sample(1)
stopwords = nltk.corpus.stopwords.words('english')
#Word usage in percentage form for each variable 

word_useage['I chi2'] = np.power((word_useage['I_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)

word_useage['E chi2'] = np.power((word_useage['E_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)



word_useage['S chi2'] = np.power((word_useage['S_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)

word_useage['N chi2'] = np.power((word_useage['N_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)



word_useage['F chi2'] = np.power((word_useage['F_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)

word_useage['T chi2'] = np.power((word_useage['T_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)



word_useage['P chi2'] = np.power((word_useage['P_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)

word_useage['J chi2'] = np.power((word_useage['J_perc'] - word_useage['all_perc']), 2) / word_useage['all_perc'].astype(np.float64)
I_words = word_useage[['I_perc', 'all_perc', 'I chi2']][(word_useage['I_perc'] > word_useage['all_perc'])].sort_values(by='I chi2', ascending=False)

E_words = word_useage[['E_perc', 'all_perc', 'E chi2']][word_useage['E_perc'] > word_useage['all_perc']].sort_values(by='E chi2', ascending=False)



S_words = word_useage[['S_perc', 'all_perc', 'S chi2']][(word_useage['S_perc'] > word_useage['all_perc'])].sort_values(by='S chi2', ascending=False)

N_words = word_useage[['N_perc', 'all_perc', 'N chi2']][word_useage['N_perc'] > word_useage['all_perc']].sort_values(by='N chi2', ascending=False)



F_words = word_useage[['F_perc', 'all_perc', 'F chi2']][(word_useage['F_perc'] > word_useage['all_perc'])].sort_values(by='F chi2', ascending=False)

T_words = word_useage[['T_perc', 'all_perc', 'T chi2']][word_useage['T_perc'] > word_useage['all_perc']].sort_values(by='T chi2', ascending=False)



P_words = word_useage[['P_perc', 'all_perc', 'P chi2']][(word_useage['P_perc'] > word_useage['all_perc'])].sort_values(by='P chi2', ascending=False)

J_words = word_useage[['J_perc', 'all_perc', 'J chi2']][word_useage['J_perc'] > word_useage['all_perc']].sort_values(by='J chi2', ascending=False)
I_keep = I_words[I_words.index.isin(list(stopwords))].head(5)

E_keep = E_words[E_words.index.isin(list(stopwords))].head(5)



S_keep = S_words[S_words.index.isin(list(stopwords))].head(5)

N_keep = N_words[N_words.index.isin(list(stopwords))].head(5)



F_keep = F_words[F_words.index.isin(list(stopwords))].head(5)

T_keep = T_words[T_words.index.isin(list(stopwords))].head(5)



P_keep = P_words[P_words.index.isin(list(stopwords))].head(5)

J_keep = J_words[J_words.index.isin(list(stopwords))].head(5)
I_keep = list(I_keep.index)

E_keep = list(E_keep.index)



S_keep = list(S_keep.index)

N_keep = list(N_keep.index)



F_keep = list(F_keep.index)

T_keep = list(T_keep.index)



P_keep = list(P_keep.index)

J_keep = list(J_keep.index)
keep = I_keep+E_keep+S_keep+N_keep+F_keep+T_keep+P_keep+J_keep



keep = set(keep)
len(keep)
stop = nltk.corpus.stopwords.words('english')
len(stop)
stopwords = []

for i in stop:

    if i in keep:

        pass

    else:

        stopwords.append(i)
len(stopwords)
train = train_df

sample = train.sample(3).reset_index(drop=True)

train.shape
#Removes links

remove_links(train, 'posts')

train.head(1)
train['posts'].replace(r'\|\|\|', r' ', regex=True, inplace=True)

train['posts'].head(1)
#Removes punchuations and set text to lowercase

train['posts'] = train['posts'].str.lower()

train['posts'] = train['posts'].apply(no_punc_num)

train['posts'].head(1)
#Tokenize the posts

train['posts'] = train['posts'].apply(tokens.tokenize)

train['posts'].head(1)
#Lemmatize the posts

train['posts'] = train['posts'].apply(lemmatized, args=(lem,))

train.head(1)
#Removes stopwords from the posts

train['posts'] = train['posts'].apply(remove_stop_words)

train.head(1)
train['posts'] = [' '.join(map(str, l)) for l in train['posts']]

train.head(1)
train['Mind']   = train['type'].apply(lambda x: x[0] == 'E').astype('int')

train['Energy'] = train['type'].apply(lambda x: x[1] == 'N').astype('int')

train['Nature'] = train['type'].apply(lambda x: x[2] == 'T').astype('int')

train['Tactics']= train['type'].apply(lambda x: x[3] == 'J').astype('int')

train = train[['Mind','Energy','Nature','Tactics','posts', 'type']]
train.head(1)
#Removes links

remove_links(test, 'posts')

test.head(1)
test['posts'].replace(r'\|\|\|', r' ', regex=True, inplace=True)

test['posts'].head(1)
#Removes punchuations and set text to lowercase

test['posts'] = test['posts'].str.lower()

test['posts'] = test['posts'].apply(no_punc_num)

test['posts'].head(1)
#Tokenize the posts

test['posts'] = test['posts'].apply(tokens.tokenize)

test['posts'].head(1)
#Lemmatize the posts

test['posts'] = test['posts'].apply(lemmatized, args=(lem,))

test.head(1)
#Removes the stopwords from the posts

test['posts'] = test['posts'].apply(remove_stop_words)

test.head(1)
test['posts'] = [' '.join(map(str, l)) for l in test['posts']]

test.head(1)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
#Vectorising using CountVectorizer

Count_vect = CountVectorizer(max_df=0.8, min_df=43,  lowercase=False)

Count_train = Count_vect.fit_transform(train['posts'])

Count_test = Count_vect.transform(test['posts'])
#Vectorising using TfidfVectorizer

Tfidf_vect =TfidfVectorizer(max_df=0.8, min_df=43, lowercase=False)

Tfidf_train = Tfidf_vect.fit_transform(train['posts'])

Tfidf_test = Tfidf_vect.transform(test['posts'])
#It seems they have exactly the same result in this case according to the results printed out.

print(f'count: {Count_train.shape}\nCount_test: {Count_test.shape}\n\nTfidf: {Tfidf_train.shape}\nTfidf_test: {Tfidf_test.shape}')
#Import libraries

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import log_loss
#We saved our id for submission purposes

subm = {}

subm['id'] = test['id'].values
np.mean(train['Mind'] == 1)
mind = LogisticRegression(C=1, solver='lbfgs')

mind.fit(Tfidf_train, train['Mind'].values)

alert()
y_probs = mind.predict_proba(Tfidf_train)

mind_pred = mind.predict(Tfidf_train)



for thresh in np.arange(0.1, 1, 0.1).round(1):

    fiddled_y = np.where(y_probs[:,1] > thresh, 1, 0)

    print(f'the loss for {thresh} is:  {log_loss(fiddled_y, train["Mind"])}')
true_mind = np.where(mind.predict_proba(Tfidf_test)[:,1] > 0.3, 1, 0)
true_mind
subm['mind'] = true_mind
np.mean(train['Energy'] == 1)

Energy  = LogisticRegression(C=1, solver='lbfgs')

Energy.fit(Tfidf_train,  train['Energy'])

alert()
y_probs = Energy.predict_proba(Tfidf_train)

Energy_pred = Energy.predict(Tfidf_train)



for thresh in np.arange(0.1, 1, 0.1).round(1):

    fiddled_y = np.where(y_probs[:,1] > thresh, 1, 0)

    print(f'the loss for {thresh} is:  {log_loss(fiddled_y, train["Energy"])}')
true_energy = np.where(Energy.predict_proba(Tfidf_test)[:,1] > 0.7, 1, 0)
true_energy
subm['energy'] = true_energy
np.mean(train['Nature'] == 1)

Nature  = LogisticRegression(C=1, solver='lbfgs')

Nature.fit(Tfidf_train,  train['Nature'])

alert()
y_probs = Nature.predict_proba(Tfidf_train)

Nature_pred = Nature.predict(Tfidf_train)



for thresh in np.arange(0.1, 1, 0.1).round(1):

    fiddled_y = np.where(y_probs[:,1] > thresh, 1, 0)

    print(f'the loss for {thresh} is:  {log_loss(fiddled_y, train["Nature"])}')
true_nature = np.where(Nature.predict_proba(Tfidf_test)[:,1] > 0.5, 1, 0)
true_nature
subm['nature'] = true_nature
np.mean(train['Tactics'] == 1)

Tactics = LogisticRegression(C=1, solver='lbfgs')

Tactics.fit(Tfidf_train, train['Tactics'])

alert()
y_probs = Tactics.predict_proba(Tfidf_train)

Tactics_pred = Tactics.predict(Tfidf_train)



for thresh in np.arange(0.1, 1, 0.1).round(1):

    fiddled_y = np.where(y_probs[:,1] > thresh, 1, 0)

    print(f'the loss for {thresh} is:  {log_loss(fiddled_y, train["Tactics"])}')
true_tactics = np.where(Tactics.predict_proba(Tfidf_test)[:,1] > 0.4, 1, 0)
true_tactics
subm['tactics'] = true_tactics
submit = pd.DataFrame(subm)
submit.sample(10)
submit.to_csv('kaggle submit.csv', index=False)