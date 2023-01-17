import pandas as pd
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import warnings
import random
from string import punctuation
import seaborn as sns

from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image

# warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize


from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score
%matplotlib inline

# Set default plot size
plt.rcParams["figure.figsize"] = (15,8)
real = pd.read_csv("../input/fake-and-real-news-dataset/True.csv")
fake = pd.read_csv("../input/fake-and-real-news-dataset/Fake.csv")
fake['Authenticity'] = 'Fake'
real['Authenticity'] = 'Real'
news_data = fake.append(real)
news_data.head()
sw = stopwords.words('english')

new_words=('’','“', '”')

for i in new_words:
    sw.append(i)


# Convert to lower case
news_data['text'] = news_data['text'].str.lower()

# Tokenizing
news_data['tokenized_text'] = news_data['text'].apply(word_tokenize)

# Remove stopwords
news_data['filtered_text'] = news_data['tokenized_text'].apply(lambda x: [item for item in x if item not in sw])

# Remove punction
news_data['filtered_text'] = news_data['filtered_text'].apply(lambda x: [item for item in x if item not in punctuation])

# Check results
print(len(news_data['text'].iloc[0]),
      len(news_data['tokenized_text'].iloc[0]),
      len(news_data['filtered_text'].iloc[0]))
news_data.head()
text = " ".join(text for text in news_data.text)

wordcloud = WordCloud(background_color="white", max_words=1000,
                      max_font_size=90, random_state=42).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


news_data_fake = news_data[news_data.Authenticity == 'Fake']
news_data_real = news_data[news_data.Authenticity == 'Real']

news_data_fake.head()
fake = news_data_fake.filtered_text.tolist()

fake_list = []
for sublist in fake:
    for item in sublist:
        fake_list.append(item)

real = news_data_real.filtered_text.tolist()

real_list = []
for sublist in real:
    for item in sublist:
        real_list.append(item)
        
all_words = news_data.filtered_text.tolist()

all_words_list = []
for sublist in all_words:
    for item in sublist:
        all_words_list.append(item)
vocab_fake = nltk.FreqDist(fake_list)
vocab_real = nltk.FreqDist(real_list)
vocab_all = nltk.FreqDist(all_words_list)

print('Fake most common words: ',vocab_fake.most_common(20),
     'Real most common words: ',vocab_real.most_common(20),
     'All most common words: ',vocab_real.most_common(20))
common_words_fake = [item[0] for item in vocab_fake.most_common(20)]
nltk.Text(fake_list[:10000]).dispersion_plot(common_words_fake)

common_words_real = [item[0] for item in vocab_real.most_common(20)]
nltk.Text(real_list[:10000]).dispersion_plot(common_words_real)
vectorizer = TfidfVectorizer(stop_words=sw,lowercase=True)
y = news_data.Authenticity
x = vectorizer.fit_transform(news_data.text)
print (x.shape)
print (y.shape)
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train,y_train)
labels = NB_classifier.predict(X_test)

mat = confusion_matrix(y_test, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('true label')
plt.ylabel('predicted label')
roc_auc_score(y_test,NB_classifier.predict_proba(X_test)[:,1])
print(classification_report(y_test,labels))
model = LogisticRegression()
model.fit(X_train, y_train)
model.score(X_test, y_test)
y_model = model.predict(X_test)

mat = confusion_matrix(y_test,y_model)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False) 
plt.xlabel('predicted value')
plt.ylabel('true value')
print(classification_report(y_test,y_model))