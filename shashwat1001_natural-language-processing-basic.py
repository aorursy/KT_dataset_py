
import numpy as np 
import pandas as pd 
import nltk
from nltk.stem import PorterStemmer
from sklearn.linear_model import LinearRegression
ps=PorterStemmer()
from nltk.corpus import stopwords
from nltk.corpus import state_union
from nltk.tokenize import word_tokenize
from nltk.tokenize import PunktSentenceTokenizer
stop=set(stopwords.words("english"))
from nltk.stem.wordnet import WordNetLemmatizer
lem = WordNetLemmatizer()
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import Pipeline
import time
import re
from sklearn import metrics
from textblob import TextBlob
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from bs4 import BeautifulSoup
from nltk.tokenize import WordPunctTokenizer
tok = WordPunctTokenizer()
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression(C = 1.1)
df=pd.read_csv('../input/datasheet/demo - Sheet1.csv')
df['sentiment'] = df['Tweet_Text'].map(lambda text: TextBlob(text).sentiment.polarity)
#nltk.word_tokenize(df['Tweet_Text'][200])
pat1 = r'@[A-Za-z0-9_]+'       
pat2 = r'https?://[^ ]+'        
combined_pat = r'|'.join((pat1, pat2)) 
www_pat = r'www.[^ ]+'        
negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not", 
                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
                "mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')

def tweet_cleaner(text):  
    soup = BeautifulSoup(text, 'lxml')   
    souped = soup.get_text()  
    try:
        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")   
    except:
        bom_removed = souped
    stripped = re.sub(combined_pat, '', bom_removed)
    stripped = re.sub(www_pat, '', stripped)
    lower_case = stripped.lower()     
    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case) 
    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)       
    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1] 
    return (" ".join(words)).strip()
nums = [0,500]
clean_tweet_texts = []
for i in range(nums[0],nums[1]):                                                          
    clean_tweet_texts.append(tweet_cleaner(df['Tweet_Text'][i]))
tokenData = nltk.word_tokenize(str(clean_tweet_texts))
fs=[w for w in tokenData if not w in stop]
for e in tokenData:
    ps.stem(e)
for r in tokenData:
    lem.lemmatize(r)
f=nltk.pos_tag(df['Tweet_Text'])
#df.info()
roundUp = df['sentiment'].apply(np.ceil) 
df.head()


token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(clean_tweet_texts)
X_train, X_test, y_train, y_test = train_test_split(text_counts, roundUp, test_size=0.3, random_state=1)
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))
tf=TfidfVectorizer(lowercase=True,stop_words='english')
text_tf= tf.fit_transform(clean_tweet_texts)
X_tr, X_te, y_tr, y_te = train_test_split(text_tf, roundUp, test_size=0.3, random_state=123)
clf = MultinomialNB().fit(X_tr, y_tr)
pre= clf.predict(X_te)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_te, pre))
X = (df['Tweet_Text'] + ' ' ).values
y = (df['sentiment'].values)
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3)
tf=TfidfVectorizer(lowercase=True,stop_words='english')
start = time.time()
t_tf= tf.fit_transform(X_train)
end = time.time()
print('Time to train vectorizer and transform training text: %0.2fs' % (end - start))
model = SGDRegressor(loss='squared_loss', penalty='l2', max_iter=5)
params = {'penalty':['none','l2','l1'],
          'alpha':[1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1]}
gs = GridSearchCV(estimator=model,
                  param_grid=params,
                  scoring='neg_mean_squared_error',
                  n_jobs=1,
                  cv=5,
                  verbose=3)
start = time.time()
gs.fit(t_tf, y_train)
end = time.time()
print('Time to train model: %0.2fs' % (end -start))
model = gs.best_estimator_
print(gs.best_params_)
print(gs.best_score_)

pipe = Pipeline([('tf',tf),('model',model)])
start = time.time()
y_pred = pipe.predict(X_test)
end = time.time()
print('Time to generate predictions on test set: %0.2fs' % (end - start))
#mean_squared_log_error(y_test, y_pred)