import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
# The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. 

# It contains one set of SMS messages in English of 5,574 messages, tagged acording being ham (legitimate) or spam.
df = pd.read_csv('../input/spam.csv', encoding='latin-1')

df.head()
df.shape
df.columns = ['label', 'msg', 'var1', 'var2', 'var3']
df['var1'].describe()
df['var2'].describe()
df['var3'].describe()
df['spam'] = np.where(df['label']=='spam', 1, 0)

df.head()
df[df['spam']==1].head() 
stopwords = set(STOPWORDS)
words = ''.join(list(df[df['spam']==1]['msg']))

spam_wc = WordCloud(background_color='white',

                    stopwords=set(STOPWORDS),

                    max_words=50,).generate(words)

plt.figure(figsize=(10,8), facecolor='k')

plt.imshow(spam_wc)

plt.axis('off')

plt.tight_layout(pad=0)
df[df['spam']==0].head()
words = ''.join(list(df[df['spam']==1]['msg']))

spam_wc = WordCloud(background_color='white',

                    stopwords=set(STOPWORDS),

                    max_words=50,).generate(words)

plt.figure(figsize=(10,8), facecolor='k')

plt.imshow(spam_wc)

plt.axis('off')

plt.tight_layout(pad=0)
X_train, X_test, y_train, y_test = train_test_split(df['msg'],

                                                    df['spam'],

                                                    random_state=0)
vect = CountVectorizer().fit(X_train)

X_train_vectorized = vect.transform(X_train)



print('every other 700th feature - ',vect.get_feature_names()[::700])

print('total number of rows/documents in of the training dataframe/corpus - ', X_train.shape)

print('number of features/words - ', len(vect.get_feature_names()))

print('shape of the vectorized train sparse matrix - ', X_train_vectorized.shape)
model = LogisticRegression()

model.fit(X_train_vectorized, y_train)

pred = model.predict(vect.transform(X_test))



score = roc_auc_score(y_test, pred)

feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()



print(score, '\n')

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:15]]))

print('Largest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:-15:-1]]))

print(model.predict(vect.transform(['you won the free call offer. call back to claim.', 

                                    'how are you'])))
print(classification_report(y_test, pred, target_names=['ham', 'spam']))



cm = confusion_matrix(pred, y_test)

df_cm = pd.DataFrame(cm, 

                     columns=np.unique(y_test), 

                     index = np.unique(y_test))

df_cm.index.name = 'Actual'

df_cm.columns.name = 'Predicted'



sns.heatmap(df_cm, 

            cmap="Blues", 

            annot=True, 

            fmt='g')
vect = TfidfVectorizer(min_df=5).fit(X_train)

X_train_vectorized = vect.transform(X_train)



model = LogisticRegression()

model.fit(X_train_vectorized, y_train)

pred = model.predict(vect.transform(X_test))



score = roc_auc_score(y_test, pred)

feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()



print(score, '\n')

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:15]]))

print('Largest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:-15:-1]]))

print(model.predict(vect.transform(['you won the free call offer. call back to claim.', 

                                    'how are you'])))
vect = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)

X_train_vectorized = vect.transform(X_train)



model = LogisticRegression()

model.fit(X_train_vectorized, y_train)

pred = model.predict(vect.transform(X_test))



score = roc_auc_score(y_test, pred)

feature_names = np.array(vect.get_feature_names())

sorted_coef_index = model.coef_[0].argsort()



print(score, '\n')

print('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:15]]))

print('Largest Coefs: \n{}\n'.format(feature_names[sorted_coef_index[:-15:-1]]))

print(model.predict(vect.transform(['you won the free call offer. call back to claim.', 

                                    'how are you'])))