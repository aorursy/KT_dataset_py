import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords

from sklearn import preprocessing

from sklearn.naive_bayes import MultinomialNB
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')

train.head(3)
lbl_enc = preprocessing.LabelEncoder()

y = lbl_enc.fit_transform(train.author.values)
ctv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), stop_words = 'english')
xtrain_ctv_all=ctv.fit_transform(train.text.values)

xtest_ctv_all=ctv.transform(test.text.values)

clf = MultinomialNB(alpha=1.0)

clf.fit(xtrain_ctv_all, y)
sub = pd.DataFrame(clf.predict_proba(xtest_ctv_all), columns=["EAP","HPL","MWS"],)

sub["id"] = test.id

cols = sub.columns.tolist()

sub = sub[cols[-1:] + cols[:-1]]

sub.to_csv("simple_spooky_sub.csv")

sub.head()