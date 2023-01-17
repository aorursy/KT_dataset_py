import pandas as pd 

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import  TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split as ttSplit

from sklearn.metrics import mean_absolute_error as MAE



import nltk 

import string

from nltk.corpus import stopwords

from nltk.tokenize import RegexpTokenizer

from nltk.stem import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer

import re

import time
train = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")



X = train['text']

y = train['target']



test = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

test = test[['id', 'text']] 



from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

start = time.time()

X = train.copy()

ps = PorterStemmer()

stop_words = set(stopwords.words("english"))



def lemmatize(text):

    tokens = word_tokenize(text)

    lematized = ''.join([lemmatizer.lemmatize(word) for word in text])

    return lematized

    

    



def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def RP(text):

    np = ''.join([c for c in text if c not in string.punctuation])

    return np



def drop_stop_words (text):

    words = word_tokenize(text)

    a = ''

    for index, txt in enumerate(words):

        a = ' '.join([w for w in words if w not in stop_words ] )

    return a

def drop_nums(text):

    dnum = ''.join([w for w in text if w not in ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']])

    return dnum



def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



def PS (text):

    words = word_tokenize(text)

    pstr = ''

    for w in words:

        pstr += ps.stem(w)+' '

    return pstr



def lower_case(text):

    return text.lower()

    



def clean_text(X):

    X = X.apply(RP)

    X = X.apply(lower_case) 

    X = X.apply(remove_emoji)

    X = X.apply(remove_URL)

    X = X.apply(drop_nums)

    X = X.apply(drop_stop_words)

    X = X.apply(PS)

    X = X.apply(lemmatize) 

    return X

#remind :change to lower case



X['text'] = clean_text(X['text'])

print('finished in {}'.format((time.time()-start)/60))

X
cv = CountVectorizer()

Xcounts = cv.fit_transform(X['text']) 



tfx = TfidfTransformer(use_idf = False).fit(Xcounts)

Xtf = tfx.transform(Xcounts)
from sklearn.linear_model import LogisticRegression
X_train, X_val, y_train, y_val = ttSplit(Xtf, y,test_size = 0.4, random_state=0)

model = LogisticRegression(random_state=0).fit(X_train, y_train)

model = LogisticRegression(random_state=0).fit(Xtf, y)

p = model.predict(X_val)
def acc(model, y_val):

    cnt = 0

    for index, y in enumerate(y_val):

        if model[index] == y:

            cnt+=1

    print('number of true positives: ',cnt)

    print("accuracy: {}%".format((cnt*100)/len(y_val))) 

    print('total number of samples: ',len(y_val))

acc(p, y_val)
test['text'] = clean_text(test['text'])



Xcnt = cv.transform(test['text'])

Xtf = tfx.transform(Xcnt)



prediction = model.predict(Xtf)



print(prediction)
df_sub = test[['id', 'text']]

s = pd.Series(prediction)



df_sub['text'] = s

df_sub.columns = ['id', 'target']

df_sub.to_csv('submission.csv', index = False)