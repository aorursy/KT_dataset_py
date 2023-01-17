import matplotlib.pyplot as plt



import numpy as np

import pandas as pd



from sklearn.metrics.pairwise import cosine_distances as cosine

from catboost import CatBoostClassifier, Pool, cv



import string

import re



import nltk

from nltk.corpus import stopwords

import hunspell



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn import metrics

from sklearn.model_selection import KFold
train = pd.read_csv('../input/tweet-sentiment-extraction/train.csv')

test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

sample_sub = pd.read_csv('../input/tweet-sentiment-extraction/sample_submission.csv')



train = train.drop(314)

train.index = np.arange(27480)
table_empty = str.maketrans({key: None for key in (string.punctuation + ' ')})

table_space = str.maketrans({key: ' ' for key in string.punctuation})



hobj = hunspell.HunSpell('/usr/share/hunspell/en_US.dic', '/usr/share/hunspell/en_US.aff')



good_smile = ['=P', '<3', ':)', '(:', ';)', '=D', ';D', 'xD', ':-D', '=}', '=]',

              '(=', '^-^', ':-*', ';-)', ':]', ':>', '*-*', '^.^', '^^']

bad_smile = ['D:', ':(', ':[', '=(', ' :/ ', '):', 'D:', ')=', '>=[', ":'(",

             ':-/', ':`(', ':-o', ':|', ':-/', ':@', ':[', ':-|', ':o', '={']
def TextCleaner(text):

    text = re.sub(r'https?:\/\/\S*', 'huperlink', text) #huperlink

    text = re.sub('[_@]\S*', 'username', text) #username

    text = re.sub('\s', ' ', text) # '\t', '\n' to ' '

    text = re.sub('[ì,í,î,ï]', 'i', text)

    text = re.sub('[À,Á,Â,Ã,Ä,Å]', 'A', text)

    text = re.sub(' w/o', ' without', text)

    text = re.sub(' b/c', ' because', text)

    text = re.sub(' w/', ' with', text)

    text = re.sub(' n ', ' and ', text) 

    text = re.sub(' u ', ' you ', text)

    text = re.sub(' r ', ' are ', text)

    text = re.sub(' u ', ' you ', text)

    text = re.sub(' U ', ' You ', text)

    text = re.sub(' ppl', ' peolpe ', text)

    text = re.sub(' pls ', ' please ', text)

    text = re.sub(' coz ', ' cause ', text)

    text = re.sub(' cuz ', ' cause ', text)

    text = re.sub(' cos ', ' cause ', text)

    text = re.sub(' wat ', ' what ', text)

    text = re.sub(' \*\*\*\* ', ' fuck ', text)

    text = re.sub(' +', ' ', text)

    text = re.sub("`", "'", text)

    text = ''.join(map(WordCleaner, 

                       re.split('(\W+)', text.lower().strip())))

    return text



def replace(text):

    st = text

    for char in set(text):

        if char in string.punctuation:

            continue

        pattern = char + '{2,}'

        st = re.sub(pattern, char, st) 

    return st

def replace2(text):

    st = text

    for char in set(text):

        if char in string.punctuation:

            continue

        pattern = char + '{3,}'

        st = re.sub(pattern, char+char, st) 

    return st



def WordCleaner(word):

    if len(word.translate(table_empty)) == 0 or hobj.spell(word):

        return word

    if hobj.spell(re.sub('0', 'o', word)):

        return re.sub('0', 'o', word)

    if hobj.spell(replace(word)):

        return replace(word)

    if hobj.spell(replace2(word)):

        return replace2(word)

    some_ideas = hobj.suggest(word)

    if (len(word) > 3 and len(some_ideas) > 0 and len(some_ideas[0].split()) < 2 

        and nltk.edit_distance(word, some_ideas[0]) < 2):

        return some_ideas[0]

    return word



def character_filter(c):

    if c == "\t": return " "

    if ord(c)<128: return c

    if c in "≠•∞™ˈʃʊʁʁiʁɑ̃ʃɔ.̃ºª¶§¡£¢ç": return "z"

    if c in "àáâãäåæ": return "a"

    if c in "èéêë": return "e"

    if c in "ìíîï": return "i"

    if c in "òóôõöōŏő": return "o"

    if c in "ùúûü": return "u"



    if c in "ÀÁÂÃÄÅ": return "A"

    if c in "ÈÉÊË": return "E"

    if c in "ÌÍÎÏ": return "I" 

    if c in "ÒÓÔÕÖŌŎŐ": return "O"

    if c in "ÙÚÛÜ": return "U"



    return "z"



def text_filter(text):

    return "".join(map(character_filter, text))





def IsOk(string):

    if hobj.spell(string) != -1:

        return True

    if hobj.spell(re.sub('0', 'o', string)) != -1:

        return True

    if hobj.spell(replace(string)) != -1:

        return True

    if hobj.spell(replace2(string)) != -1:

        return True

    for r in range(3,len(string)-3):

        if hobj.spell(string[:r]) and IsOk(string[r:]):

            return True

    some_ideas = hobj.suggest(string)

    if len(some_ideas) > 0 and nltk.edit_distance(string, some_ideas[0]) < 2:

        return True

    return False



def WordTransform(word):

    word = text_filter(word)

    if len(word) == 0 or word[0] == '@' or word[0] == '_' or word.find('http') != -1:

        return ''

    a = word.find('****') 

    if a != -1:

        word = word[:a] + 'fuck' + word[:(a+4)]

    word = word.translate(table_empty).lower()

    if len(word) == 0 or hobj.spell(word):

        return word

    if hobj.spell(re.sub('0', 'o', word)):

        return re.sub('0', 'o', word)

    if hobj.spell(replace(word)):

        return replace(word)

    if hobj.spell(replace2(word)):

        return replace2(word)

    for r in range(3,len(word)-3):

        if hobj.spell(word[:r]) and IsOk(word[r:]):

            return word[:r] + ' ' + WordTransform(word[r:])

    some_ideas = hobj.suggest(word)

    if len(some_ideas) > 0 and nltk.edit_distance(word, some_ideas[0]) < 2:

        return some_ideas[0]

    return word
train['good'] = np.array(list(map(lambda text: sum([text.count(smile) for smile in good_smile]), 

                  train['text'])))

train['bad'] = np.array(list(map(lambda text: sum([text.count(smile) for smile in bad_smile]), 

                  train['text'])))

test['good'] = np.array(list(map(lambda text: sum([text.count(smile) for smile in good_smile]), 

                  test['text'])))

test['bad'] = np.array(list(map(lambda text: sum([text.count(smile) for smile in bad_smile]), 

                  test['text'])))



train['cleaned_text'] = list(map(TextCleaner, train['text']))

                             

test['cleaned_text'] = list(map(TextCleaner, test['text']))



train['exclamation'] = np.array(list(map(lambda x: x.count('!'), train['text'])))

test['exclamation'] = np.array(list(map(lambda x: x.count('!'), test['text'])))



train['CapsLock'] = np.array(list(map(lambda text: np.array(list(map(str.isupper, text))).mean(), train['text'])))

test['CapsLock'] = np.array(list(map(lambda text: np.array(list(map(str.isupper, text))).mean(), test['text'])))



train['lenght'] = np.array(list(map(len, train['cleaned_text'])))

test['lenght'] = np.array(list(map(len, test['cleaned_text'])))



train['missed'] = np.array(list(map(lambda x: len(x) - x.count(' '),

                  train['text']))) - np.array(list(map(lambda x: len(x) - x.count(' '), 

                                                       train['cleaned_text'])))

test['missed'] = np.array(list(map(lambda x: len(x) - x.count(' '),

                  test['text']))) - np.array(list(map(lambda x: len(x) - x.count(' '), 

                                                       test['cleaned_text'])))



train['url'] = np.array(list(map(lambda x: x.find('http') != -1, train['text']))).astype(int)

test['url'] = np.array(list(map(lambda x: x.find('http') != -1, test['text']))).astype(int)
def count_len(text):

    words = text.split(' ')

    return len(list(filter(lambda word: len(word.translate(table_empty)) > 1 and word.count('http://') == 0 and

                    word[0] != '@' and word[0] != '_' and word[0] != '#',

                    words))) + 1



fraction = np.array([[count_len(x['selected_text']),

                     count_len(x['text'])] for _, x in train.iterrows()])



train['take_all'] = (fraction[:,0] >= fraction[:,1]).astype(int)
train[:5]
bert_clean_embed = np.load('../input/embedded-twits/clean_embed.npy')

bert_selected_embed = np.load('../input/embedded-twits/selected_embed.npy')

bert_test_embed = np.load('../input/embedded-twits/test_embed.npy')



embed_target =((train['sentiment'] == 'positive').astype(int) + 2 * (train['sentiment'] == 'negative').astype(int)).values



lda = LinearDiscriminantAnalysis(solver='eigen')

lda.fit(bert_selected_embed, embed_target)



train = pd.concat([train, 

                   pd.DataFrame(lda.transform(bert_clean_embed), 

                                columns=['lda_bert_0', 'lda_bert_1', ])], 

                  axis=1, sort=False)



test = pd.concat([test, 

                   pd.DataFrame(lda.transform(bert_test_embed), 

                                columns=['lda_bert_0', 'lda_bert_1', ])], 

                  axis=1, sort=False)
((lda.predict(bert_clean_embed) ==  embed_target) == train['take_all']).mean()
col = ['cleaned_text', 

       'sentiment',

       'exclamation', 

       'CapsLock',

       'missed',

       'lenght',

       'good',

       'bad',

       'lda_bert_0', 'lda_bert_1',

       'url',

      ]



cv_dataset = Pool(data=train[col],

                  label=train['take_all'],

                  cat_features=['sentiment'],

                  text_features=['cleaned_text'])
rkf = KFold(n_splits=5, random_state=42)

for cv_train, cv_test in rkf.split(np.arange(train.shape[0])):

    cat = CatBoostClassifier(n_estimators=500,

                         max_depth = 8,

                         task_type = 'GPU',

                         verbose = 0

                        )

    cat.fit(train.loc[cv_train, col], train.loc[cv_train, 'take_all'],

        cat_features=['sentiment'],

        text_features=['cleaned_text'])

    print('accuracy: ' + str((cat.predict(train.loc[cv_test, col]) == train.loc[cv_test, 'take_all']).mean()))
def rr(df, target):

    for _, x in df.iterrows():

        for word in re.split('(\W+)', x['text'].strip()):

            y = x.copy()

            y['place'] = x['text'].find(word)

            y['word'] = WordTransform(word).lower()

            if len(y['word'].translate(table_empty)) < 3:

                continue

            y['missed'] = len(word) - len(y['word'])

            y['CapsLock'] = sum(list(map(str.isupper, y['word'])))

            if target:

                y['target'] = int(x['selected_text'].find(word) != -1)

                del y['selected_text']

            del y['text']

            yield y
selected_train = pd.DataFrame(rr(train, True))

selected_test = pd.DataFrame(rr(test, False))
#from collections import Counter

#words_train = Counter(selected_train['word'])

#words_test = Counter(selected_test['word'])

#embed_train = dict(zip(list(words_train), lda.transform(model.encode(list(words_train), batch_size=128))))

#embed_test = dict(zip(list(words_test), lda.transform(model.encode(list(words_test), batch_size=128))))

#selected_train['bert_0'] = [embed_train[word][0] for word in selected_train['word']]

#selected_train['bert_1'] = [embed_train[word][1] for word in selected_train['word']]

#selected_test['bert_0'] = [embed_test[word][0] for word in selected_test['word']]

#selected_test['bert_1'] = [embed_test[word][1] for word in selected_test['word']]
cat = CatBoostClassifier(n_estimators=2000,

                         max_depth = 8,

                         thread_count=6,

                         task_type = 'GPU',

                         verbose = 0)



col = ['cleaned_text', 'sentiment',

                            'CapsLock',

                            'missed', 

                            'lenght',

      #                     'bert_0', 'bert_1',

      ]



cat.fit(selected_train[col], 

        selected_train['target'],

        cat_features=['sentiment',],

        text_features=['cleaned_text'])
def sentiment_ext(i, df):

    #emb = lda.transform(np.array(model.encode(df.loc[i,'cleaned_text'].split(), batch_size=16)))

    tt = pd.concat([df.iloc[[i]][col]*emb.shape[0]])

    #tt['bert_0'] = emb[:,0]

    #tt['bert_1'] = emb[:,1]

    res = cat.predict_proba(tt)[:,1]

    words = df.loc[i, 'text'].split()

    answer = ''

    if (res < 0.5).mean() == 1:

        answer = words[np.argmax(res)]

    else:

        good = np.where(res >= 0.5)[0]

        answer = ' '.join(words[i] for i in range(good[0], good[-1] + 1))

    return answer
test['selected_text'] = [sentiment_ext(i, test) for i in range(test.shape[0])]
test[['textID','selected_text']].to_csv('submission.csv',index=False)