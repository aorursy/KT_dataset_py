import numpy as np

import pandas as pd

import string

import matplotlib.pyplot as plt

import seaborn as sns

import nltk

import sys

from textblob import TextBlob 



import re

import warnings

warnings.filterwarnings('ignore')



from wordcloud import WordCloud, STOPWORDS 

from IPython.display import display



import spacy



# Create an empty model

nlp = spacy.load('en')


train = pd.read_csv('../input/nlp-getting-started/train.csv')

test = pd.read_csv("../input/nlp-getting-started/test.csv")
print('train:')

print(train.shape)

train.head()
print('test:')

print(test.shape)

test.head()
print('train:')

df_null = pd.DataFrame(train.isnull().sum())

df_null = df_null.rename(columns={0:'Number of null values'})

df_null['Percentage null values'] = round(train.isnull().sum()/train.id.count()*100,2)

df_null
print('test:')

df_null = pd.DataFrame(test.isnull().sum())

df_null = df_null.rename(columns={0:'Number of null values'})

df_null['Percentage null values'] = round(test.isnull().sum()/test.id.count()*100,2)

df_null
train.info()
test.info()
print('- Real')

display(train[train['target']==1].groupby('keyword')['keyword'].agg(['count']))

print('- Fake')

display(train[train['target']==0].groupby('keyword')['keyword'].agg(['count']))

print('- Real')

display(train[train['target']==1].groupby('location')['location'].agg(['count']))

print('- Fake')

display(train[train['target']==0].groupby('location')['location'].agg(['count']))
Per_Real = round((train[train.target==1].target.count()/train.target.count())*100,2)

Per_Fake = round((train[train.target==0].target.count()/train.target.count())*100,2)



table_data=[

    ["Percentage of Reals (1)", Per_Real],

    ["Percentage of Fakes (0)", Per_Fake]

]



fig = plt.figure()

# definitions for the axes

left, width = 0.10, 1.5

bottom, height = 0.1, .8

bottom_h = left_h = left + width + 0.02



rect_cones = [left, bottom, width, height]

rect_box = [left_h, bottom, 0.17, height]



# plot

ax1 = plt.axes(rect_cones)

train.groupby('target')['target'].agg(['count']).plot.bar(ax=ax1)

plt.title('Frequency of target')



ax2 = plt.axes(rect_box)

my_table = ax2.table(cellText = table_data, loc ='right')

my_table.set_fontsize(40)

my_table.scale(4,4)

ax2.axis('off')

plt.show()
num_caracters = []

num_words = [] 

for i in train.text:

    aux = nlp(i)

    num_words.append(len(aux))

    num_caracters.append(len(i))
train['num_caracters'] = num_caracters



train['num_words'] = num_words
fig, axes = plt.subplots(nrows=1,ncols=1, figsize=(12,5))

train[train.target == 0].num_caracters.plot.hist(bins=150,  ax = axes, label = 'Fake', color = 'blue')

train[train.target == 1].num_caracters.plot.hist(bins=150,  ax = axes, label = 'Real', color = 'orange')

plt.xlim(0,max(train.num_caracters))

axes.legend()

plt.title('# Characters tweeds Histogram')

plt.show()
fig, axes = plt.subplots(nrows=1,ncols=2, figsize=(12,5))

train[train.target == 0].num_words.plot.hist(bins=40,  ax = axes[0], label = 'Fake', color='blue')

axes[0].set_title('# Words tweeds Histogram (Fake)')

train[train.target == 1].num_words.plot.hist(bins=40,  ax = axes[1], label = 'Real', color = 'orange')

plt.xlim(0,max(train.num_words))

plt.title('# Words tweeds Histogram (Real)')

plt.show()

print('mean(# Words tweeds (fake)){}'.format(round(train[train.target == 0].num_words.mean(),2)))

print('std(# Words tweeds(fake)){}'.format(round(train[train.target == 0].num_words.std(),2)))

print('mean(# Words tweeds (real)){}'.format(round(train[train.target == 1].num_words.mean(),2)))

print('std(# Words tweeds(real)){}'.format(round(train[train.target == 1].num_words.std(),2)))
#creating a list where we classify the words which are stop or not.



def classify_word(serie):

    '''you pass a serie from dataframe that contains words

        returns two list: 1. non stop words and 2. stop words

    '''

    not_stop_list = []

    stop_list = []

    for line in serie:

        for token in nlp(line):

            if token.text != " ":

                if token.is_stop:

                    stop_list.append(token)

                else:

                    not_stop_list.append(token)

    

    return(not_stop_list, stop_list)
train.text = train.text.str.lower() # convert all words in lowercase
not_stopwords_real_list, stopwords_real_list = classify_word(train[train.target==1].text)

#not_stopwords_real_list = [str(x).lower() for x in not_stopwords_real_list] # convert everything in lowercase

#stopwords_real_list = [str(x).lower() for x in stopwords_real_list] #convert everything in lowercase

not_stopwords_fake_list, stopwords_fake_list = classify_word(train[train.target==0].text)

#not_stopwords_fake_list = [str(x).lower() for x in not_stopwords_fake_list] # convert everything in lowercase

#stopwords_fake_list = [str(x).lower() for x in stopwords_fake_list] #convert everything in lowercase
class count_words():

    '''

        It is a class that contains two function. Both of functions return dict with the frequency of the list

        or token you pass. Example return: key: word and value: number of repetitions of the word. 

    '''

    

    def __init__(self, list_pass):

        self.list_pass = list_pass

    

    def count_token(list_pass):

        '''you pass a token'''

        count = {}

        for word in list_pass:

            if word.text in count :

                count[word.text] += 1

            else:

                count[word.text] = 1

        return(count)  

    def count_list(list_pass):

        '''you pass a str'''

        count = {}

        for word in list_pass:

            if word in count :

                count[word] += 1

            else:

                count[word] = 1

        return(count)
#counting words 

stopwords_real_dict = count_words.count_token(stopwords_real_list)

stopwords_fake_dict = count_words.count_token(stopwords_fake_list)
#sorting dictionary by values

stopwords_real_dict = {k:stopwords_real_dict[k] for k in sorted(stopwords_real_dict, key=stopwords_real_dict.get, reverse = True)}

stopwords_fake_dict = {k:stopwords_fake_dict[k] for k in sorted(stopwords_fake_dict, key=stopwords_fake_dict.get, reverse = True)}



#getting 20th first

stopwords_fake_dict_20 = {k: stopwords_fake_dict[k] for k in list(stopwords_fake_dict)[:20]}

stopwords_real_dict_20 = {k: stopwords_real_dict[k] for k in list(stopwords_real_dict)[:20]}
fig, axes = plt.subplots(nrows=2,ncols=1, figsize=(12,5))

plt.subplots_adjust(hspace = .4)

axes[0].bar(stopwords_fake_dict_20.keys(), stopwords_fake_dict_20.values(), width = .5, color = 'b')

axes[0].set_title('20th stopwords more populars from Fake tweets')

axes[1].bar(stopwords_real_dict_20.keys(), stopwords_real_dict_20.values(), width = .5, color = 'orange')

axes[1].set_title('20th stopwords more populars from Real tweets')

plt.show()
def punctuation_funct(serie):

    ''' 

        function you pass a serie from pandas and

        it returns a list which contains punctuation symbols

    '''

    punctuation_list = []

    

    for line in serie:

        line = line.split(' ')

        for word in line:

            if word in string.punctuation:

                punctuation_list.append(word)

    return(punctuation_list)

punctuation_fake_list = punctuation_funct(train[train.target == 0].text)

punctuation_real_list = punctuation_funct(train[train.target == 1].text)



# dictionary with the frequency

punctuation_fake_dict = count_words.count_list(punctuation_fake_list)

punctuation_real_dict = count_words.count_list(punctuation_real_list)

#sorting dictionary by values

punctuation_real_dict = {k:punctuation_real_dict[k] for k in sorted(punctuation_real_dict, key=punctuation_real_dict.get, reverse = True)}

punctuation_fake_dict = {k:punctuation_fake_dict[k] for k in sorted(punctuation_fake_dict, key=punctuation_fake_dict.get, reverse = True)}

fig, axes = plt.subplots(nrows=2,ncols=1, figsize=(12,5))

plt.subplots_adjust(hspace = .4)

axes[0].bar(punctuation_fake_dict.keys(), punctuation_fake_dict.values(), width = .5, color = 'b')

axes[0].set_title('Punctuation from Fake tweets')

axes[1].bar(punctuation_real_dict.keys(), punctuation_real_dict.values(), width = .5, color = 'orange')

axes[1].set_title('Punctuation from Real tweets')

plt.show()
class removing():

    def __init__(sel,texto):

        self.text = texto

    def remove_url(texto):

        return(re.sub(r'http://\S+|https://\S+','', texto))

    def remove_emoji(texto):

        emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

        return emoji_pattern.sub(r'', texto)

    def remove_punctuation(texto):

        return(re.sub(r'[^\w\s]','',texto))
def spelling_correcting(texto):

    return(str(TextBlob(texto).correct()))
train['text_not_'] = train.text.apply(lambda x:removing.remove_url(x))

train['text_not_'] = train.text_not_.apply(lambda x: removing.remove_punctuation(x))

train['text_not_'] = train.text_not_.apply(lambda x: removing.remove_emoji(x))



# doing the same with test set

test['text_not_'] = test.text.apply(lambda x:removing.remove_url(x))

test['text_not_'] = test.text_not_.apply(lambda x: removing.remove_punctuation(x))

test['text_not_'] = test.text_not_.apply(lambda x: removing.remove_emoji(x))
#converting all words in lower

train['text_not_'] = train.text_not_.str.lower()

#converting all words in lower

test['text_not_'] = test.text_not_.str.lower()
#train['text_not_correct'] = train.text_not_.apply(lambda x: spelling_correcting(x))

#test['text_not_correct'] = test.text_not_.apply(lambda x: spelling_correcting(x))
# We are doing this because the step before takes a long time to get the values. Therefore, it has been executed

# and saved the results.



#train.to_csv('..input/set-modifies/train_aux.csv', index=None, header=True)

#test.to_csv('..input/set-modifies/test_aux.csv', index=None, header=True)





train = pd.read_csv('../input/set-modifies/train_aux.csv')

test = pd.read_csv('../input/set-modifies/test_aux.csv')
not_stop_list_real, stop_list_real = classify_word(train[train.target == 1].text_not_correct)

not_stop_list_fake, stop_list_fake = classify_word(train[train.target == 0].text_not_correct)
def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        collocations = False,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

    ).generate(str(data))

    return(wordcloud)

    
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))

#real

axes[0].axis('off')

axes[0].imshow(show_wordcloud(not_stop_list_real))

axes[0].set_title('Words from Real text without stop words', fontsize=20)

#fake

axes[1].axis('off')

axes[1].imshow(show_wordcloud(not_stop_list_fake))

axes[1].set_title('Words from Fake text without stop words', fontsize=20)

plt.show()
not_stop_dict_real = count_words.count_token(not_stop_list_real)

not_stop_dict_fake = count_words.count_token(not_stop_list_fake)
def isdict1_notdict2(dict1, dict2):

    ''' Passing two dictionaries, 

        Returning list with all elements keys and frequency the values 

        Example: {d:5}=[d,d,d,d,d]

    '''

    dict_aux = {}

    for key, value in dict1.items():

        if key not in dict2.keys():

            dict_aux[key] = value

            

    list_aux = []

    for key, value in dict_aux.items():

        list_aux = list_aux + ([key] * value)

    return(list_aux)
real_not_fake_list = isdict1_notdict2(not_stop_dict_real,not_stop_dict_fake) #it is in real and not in fake

fake_not_real_list = isdict1_notdict2(not_stop_dict_fake, not_stop_dict_real) #it is in fake and not in real
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))

#real

axes[0].axis('off')

axes[0].imshow(show_wordcloud(' '.join(sorted(real_not_fake_list))))

axes[0].set_title('Words are in Real Not in Fake (not stopwords)', fontsize=20)

#fake

axes[1].axis('off')

axes[1].imshow(show_wordcloud(' '.join(sorted(fake_not_real_list))))

axes[1].set_title('Words are in Fake Not in Real (not stopwords)', fontsize=20)

plt.show()

import numpy as np



np.random.seed(0)

train = train.reindex(np.random.permutation(train.index))





X_train = train.loc[:6852, 'text_not_correct'].values

y_train = train.loc[:6852, 'target'].values

X_test = train.loc[6852:, 'text_not_correct'].values

y_test = train.loc[6852:, 'target'].values



from nltk.stem.porter import PorterStemmer



porter = PorterStemmer()



def tokenizer(text):

    return text.split()





def tokenizer_porter(text):

    return [porter.stem(word) for word in text.split()]



from nltk.corpus import stopwords



stop = stopwords.words('english')



from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import GridSearchCV



tfidf = TfidfVectorizer(strip_accents=None,

                        lowercase=False,

                        preprocessor=None)



param_grid = [{'vect__ngram_range': [(1, 1)],

               'vect__stop_words': [stop, None],

               'vect__tokenizer': [tokenizer, tokenizer_porter],

               'clf__penalty': ['l1', 'l2'],

               'clf__C': [1.0, 10.0, 100.0]},

              {'vect__ngram_range': [(1, 1)],

               'vect__stop_words': [stop, None],

               'vect__tokenizer': [tokenizer, tokenizer_porter],

               'vect__use_idf':[False],

               'vect__norm':[None],

               'clf__penalty': ['l1', 'l2'],

               'clf__C': [1.0, 10.0, 100.0]},

              ]



lr_tfidf = Pipeline([('vect', tfidf),

                     ('clf', LogisticRegression(random_state=0))])



gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,

                           scoring='accuracy',

                           cv=5,

                           verbose=1,

                           n_jobs=-1)



gs_lr_tfidf.fit(X_train, y_train)



clf = gs_lr_tfidf.best_estimator_

print('Test Accuracy: %.3f' % clf.score(X_test, y_test))

y_pred = gs_lr_tfidf.predict(test.text_not_)
submission = pd.DataFrame({

        "id":test['id'],

        "target": y_pred

    })