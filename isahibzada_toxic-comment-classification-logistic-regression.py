import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
toxic = pd.read_csv('../input/train.csv')
toxic.head()
toxic.shape
toxic = toxic.set_index('id')
toxic.head()
toxic['text length'] = toxic['comment_text'].apply(len)
sns.distplot(a=toxic['text length'],bins=30)
fig,ax = plt.subplots(2,3,figsize=(16,10))
ax1,ax2,ax3,ax4,ax5,ax6 = ax.flatten()
sns.countplot(toxic['toxic'],palette= 'magma',ax=ax1)
sns.countplot(toxic['severe_toxic'], palette= 'viridis',ax=ax2)
sns.countplot(toxic['obscene'], palette= 'Set1',ax=ax3)
sns.countplot(toxic['threat'], palette= 'viridis',ax = ax4)
sns.countplot(toxic['insult'], palette = 'magma',ax=ax5)
sns.countplot(toxic['identity_hate'], palette = 'Set1', ax = ax6)

toxic.hist(column='text length', by='toxic', bins=50,figsize=(12,4))
toxic.hist(column='text length', by='severe_toxic', bins=50,figsize=(12,4))
import string
from nltk.corpus import stopwords
toxic['comment_text'].fillna("unknown", inplace=True)


toxic.head()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
tfidf_vec = TfidfVectorizer(max_df=0.7,stop_words='english')

X = toxic['comment_text']
y = toxic['toxic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_toxic = LogisticRegression()
log_toxic.fit(X_train_vec,y_train)

predictions = log_toxic.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

X = toxic['comment_text']
y = toxic['severe_toxic']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_stoxic = LogisticRegression()
log_stoxic.fit(X_train_vec,y_train)

predictions = log_stoxic.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

X = toxic['comment_text']
y = toxic['obscene']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_obscene = LogisticRegression()
log_obscene.fit(X_train_vec,y_train)

predictions = log_obscene.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


X = toxic['comment_text']
y = toxic['threat']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_threat = LogisticRegression()
log_threat.fit(X_train_vec,y_train)

predictions = log_threat.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

X = toxic['comment_text']
y = toxic['insult']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_insult = LogisticRegression()
log_insult.fit(X_train_vec,y_train)

predictions = log_insult.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

X = toxic['comment_text']
y = toxic['identity_hate']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train_vec = tfidf_vec.fit_transform(X_train)
X_test_vec = tfidf_vec.transform(X_test)

log_ihate = LogisticRegression()
log_ihate.fit(X_train_vec,y_train)

predictions = log_ihate.predict(X_test_vec)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))

test = pd.read_csv('../input/test.csv')
test.head()
test1 = test['comment_text']
test1_vec = tfidf_vec.transform(test1)
prob_toxic = log_toxic.predict_proba(test1_vec)
prob_stoxic = log_stoxic.predict_proba(test1_vec)
prob_obscene = log_obscene.predict_proba(test1_vec)
prob_threat = log_obscene.predict_proba(test1_vec)
prob_insult = log_insult.predict_proba(test1_vec)
prob_ihate = log_ihate.predict_proba(test1_vec)

df1 = pd.DataFrame(prob_toxic[:,1],columns={'toxic'})
df2 = pd.DataFrame(prob_stoxic[:,1],columns={'severe_toxic'})
df3 = pd.DataFrame(prob_obscene[:,1],columns={'obscene'})
df4 = pd.DataFrame(prob_threat[:,1],columns={'threat'})
df5 = pd.DataFrame(prob_insult[:,1],columns={'insult'})
df6 = pd.DataFrame(prob_ihate[:,1],columns={'identity_hate'})
df7 = pd.concat([test['id'],df1,df2,df3,df4,df5,df6],axis=1)
df7.head()
df7.set_index('id',inplace=True)
df7.to_csv(path_or_buf='submission.csv')
df7.head()

