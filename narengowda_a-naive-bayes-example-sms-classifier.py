import os
print(os.listdir("../input"))

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
%matplotlib inline

df = pd.read_csv('../input/spam.csv', encoding="cp1252")
df.head()
sms = pd.DataFrame()
sms['target'] = df['v1']
sms['sms'] = df['v2']
sms.tail()
print(sms['target'].value_counts())
sms.count()
sms['sms'].head()
from nltk.corpus import stopwords

# Remove punctuations
sms['sms'] = sms['sms'].str.replace('[^\w\s]','')

# Remove numbers
sms['sms'] = sms['sms'].str.replace('\d+', ' ')

# Remove stop words and lower case
sms['sms'] = sms['sms'].apply(lambda x: ' '.join([j.lower() for j in x.split(' ') if j not in stopwords.words('english')]))

sms['sms'].head()
from nltk import word_tokenize
import matplotlib.pyplot as plt

spam_word_cloud = WordCloud(max_words = 30).generate("".join(sms.loc[sms['target'] == 'spam']['sms']))
ham_word_cloud = WordCloud(max_words = 30).generate("".join(sms.loc[sms['target'] == 'ham']['sms']))

# Display the generated image
plt.imshow(spam_word_cloud)
plt.axis("off")
plt.suptitle('Spam words cloud', fontsize=20)
plt.show()

plt.imshow(ham_word_cloud)
plt.suptitle('Ham words cloud', fontsize=20)
plt.axis("off")
plt.show()
sms['length']=sms['sms'].apply(len)
sms.head()
sms['length'].plot(bins=50,kind='hist')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(sms['sms'], sms['target'], random_state=0)


vectorizer = CountVectorizer()
VX = vectorizer.fit_transform(X_train)

print(vectorizer.get_feature_names()[:10])

X_train_vectorized = VX.toarray()
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB(alpha=0.1)
model.fit(X_train_vectorized, y_train)
from sklearn.metrics import accuracy_score

y_pred = model.predict(vectorizer.transform(X_test))

print('Accuracy: %.2f%%' % (accuracy_score(y_test, y_pred) * 100))



for i, v in enumerate(zip(y_test, X_test)):
    if v[0] != y_pred[i]:
        print(">>>> Actual {} -- predicted -- {}".format(v[0], y_pred[i]))
        print(v[1])

