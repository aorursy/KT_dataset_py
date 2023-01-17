import numpy as np

import pandas as pd



pd.set_option('display.max_colwidth', 100)
data = pd.read_csv('../input/smsspamdetection/SMSSpamCollection.tsv', sep='\t', header=None)

data.columns = ['label', 'body_text']

data.head()
data.tail()
#Remove Punctuation



import string



def remove_punct(text):

    text_nopunct = "".join(char for char in text if char not in string.punctuation)

    return text_nopunct



data['body_text_clean'] = data['body_text'].apply(lambda x : remove_punct(x))



data.head()
print(string.punctuation)
#tokenize



import re



def tokenize(text):

    token = re.split('\W+', text)

    return token



data['body_text_token'] = data['body_text_clean'].apply(lambda x : tokenize(x))

data.head()
#Remove Stopwords



import nltk

nltk.download('stopwords')



stopword = nltk.corpus.stopwords.words('english')



def remove_stopwords(tokenize_list):

    text = [word for word in tokenize_list if word not in stopword]

    return text



data['body_text_nostop'] = data['body_text_token'].apply(lambda x : remove_stopwords(x))

data.head()
print(stopword)

print(len(stopword))
#Stemming



ps = nltk.PorterStemmer()



def stemmer(tokenize_text):

    text = [ps.stem(word) for word in tokenize_text]

    return text



data['body_text_stem'] = data['body_text_nostop'].apply(lambda x : stemmer(x))

data.head()
#lemmatize



wn = nltk.WordNetLemmatizer()



def lemmatize(tokenize_text):

    text = [wn.lemmatize(word) for word in tokenize_text]

    return text



data['body_text_lemma'] = data['body_text_nostop'].apply(lambda x : lemmatize(x))

data.head()
# Apply CountVectorizer



from sklearn.feature_extraction.text import CountVectorizer



stopwords = nltk.corpus.stopwords.words('english')

ps = nltk.PorterStemmer()



df = pd.read_csv('../input/smsspamdetection/SMSSpamCollection.tsv', sep='\t', header=None)

df.columns = ['label', 'body_text']

df.head()



def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens = re.split('\W+', text)

    text = [ps.stem(word) for word in tokens if word not in stopwords]

    return text



count_vect = CountVectorizer(analyzer=clean_text)

X_counts = count_vect.fit_transform(df['body_text'])

print(X_counts.shape)

print(count_vect.get_feature_names())
X_counts_df = pd.DataFrame(X_counts.toarray())

X_counts_df
X_counts_df.columns = count_vect.get_feature_names()

X_counts_df
def clean_txt(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens = re.split('\W+', text)

    text = " ".join([ps.stem(word) for word in tokens if word not in stopwords])

    return text



df['clean_text'] = df['body_text'].apply(lambda x : clean_txt(x))

df.head()
from sklearn.feature_extraction.text import CountVectorizer



ngram_vect = CountVectorizer(ngram_range=(2,2))

X_counts = ngram_vect.fit_transform(df['clean_text'])

print(X_counts.shape)

print(ngram_vect.get_feature_names())
X_counts_df = pd.DataFrame(X_counts.toarray())

X_counts_df.columns = ngram_vect.get_feature_names()

X_counts_df
from sklearn.feature_extraction.text import TfidfVectorizer



tfidf_vect = TfidfVectorizer(analyzer=clean_text)

X_tfidf = tfidf_vect.fit_transform(df['body_text'])

print(X_tfidf.shape)

print(tfidf_vect.get_feature_names())
X_tfidf_df = pd.DataFrame(X_tfidf.toarray())

X_tfidf_df.columns = tfidf_vect.get_feature_names()

X_tfidf_df
data = pd.read_csv('../input/smsspamdetection/SMSSpamCollection.tsv', sep='\t', header=None)

data.columns = ['label', 'body_text']

data.head()
#Create feature for text message length

data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))



data.head()
#Create feature for % of text that is punctuation

def count_punct(text):

    count = sum([1 for char in text if char in string.punctuation])

    return round(count/(len(text) - text.count(" ")), 3)*100



data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))



data.head()
#Evaluate created features



from matplotlib import pyplot

%matplotlib inline
bins = np.linspace(0,200,40)



pyplot.hist(data[data['label']=='spam']['body_len'], bins, alpha=0.5, label='spam')

pyplot.hist(data[data['label']=='ham']['body_len'], bins, alpha=0.5, label='ham')

pyplot.legend(loc='upper left')

pyplot.show()
bins = np.linspace(0, 50, 40)



pyplot.hist(data[data['label']=='spam']['punct%'], bins, alpha=0.5, label='spam')

pyplot.hist(data[data['label']=='ham']['punct%'], bins, alpha=0.5, label='ham')

pyplot.legend(loc='upper right')

pyplot.show()
bins = np.linspace(0,200,40)



pyplot.hist(data['body_len'], bins)

pyplot.title("Body Length Distribution")

pyplot.show()
bins = np.linspace(0,50,40)



pyplot.hist(data['punct%'], bins)

pyplot.title("Punctuation % Distribution")

pyplot.show()
data = pd.read_csv('../input/smsspamdetection/SMSSpamCollection.tsv', sep='\t', header=None)

data.columns = ['label', 'body_text']

data.head()
data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))

data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))

data.head()
def clean_text(text):

    text = "".join([word.lower() for word in text if word not in string.punctuation])

    tokens = re.split('\W+', text)

    text = [ps.stem(word) for word in tokens if word not in stopwords]

    return text

from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(data[['body_text', 'body_len', 'punct%']], data['label'], test_size=0.2)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
tfidf_vect = TfidfVectorizer(analyzer=clean_text)

tfidf_vect_fit = tfidf_vect.fit(X_train['body_text'])



tfidf_train = tfidf_vect_fit.transform(X_train['body_text'])

tfidf_test = tfidf_vect_fit.transform(X_test['body_text'])



X_train_vect = pd.concat([X_train[['body_len', 'punct%']].reset_index(drop=True), 

           pd.DataFrame(tfidf_train.toarray())], axis=1)

X_test_vect = pd.concat([X_test[['body_len', 'punct%']].reset_index(drop=True), 

           pd.DataFrame(tfidf_test.toarray())], axis=1)



X_train_vect.head()
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import precision_recall_fscore_support as score

import time
rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=-1)



start = time.time()

rf_model = rf.fit(X_train_vect, y_train)

end = time.time()

fit_time = (end - start)



start = time.time()

y_pred = rf_model.predict(X_test_vect)

end = time.time()

pred_time = (end - start)



precision, recall, fscore, train_support = score(y_test, y_pred, pos_label='spam', average='binary')

print('Fit time: {} / Predict time: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(

    round(fit_time, 3), round(pred_time, 3), round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))
gb = GradientBoostingClassifier(n_estimators=150, max_depth=11)



start = time.time()

gb_model = gb.fit(X_train_vect, y_train)

end = time.time()

fit_time = (end - start)



start = time.time()

y_pred = gb_model.predict(X_test_vect)

end = time.time()

pred_time = (end - start)



precision, recall, fscore, train_support = score(y_test, y_pred, pos_label='spam', average='binary')

print('Fit time: {} / Predict time: {} ---- Precision: {} / Recall: {} / Accuracy: {}'.format(

    round(fit_time, 3), round(pred_time, 3), round(precision, 3), round(recall, 3), round((y_pred==y_test).sum()/len(y_pred), 3)))