import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

import wordcloud



filename = '../input/naivebayesleariningsamples/spam_ham.csv'



print('setup complete!')
df = pd.read_csv(filename, encoding='latin-1')



df
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

df = df.rename(columns={"v1":"label", "v2":"text"})
df.label.value_counts()
df.label.value_counts().plot.bar();
df['length'] = df['text'].apply(len)



df.head(10)
df.hist(column='length', by='label', bins=50, figsize=(12, 4))

plt.xlim(-20, 550)
def show_wordcloud(data_spam_or_ham, title):

    text = ' '.join(data_spam_or_ham['text'].astype(str).tolist())

    stopwords = set(wordcloud.STOPWORDS)

    

    fig_wordcloud = wordcloud.WordCloud(stopwords=stopwords,background_color='lightgrey',

                    colormap='viridis', width=800, height=600).generate(text)

    

    plt.figure(figsize=(10,7), frameon=True)

    plt.imshow(fig_wordcloud)  

    plt.axis('off')

    plt.title(title, fontsize=20 )

    plt.show()
data_ham  = df[df['label'] == 'ham'].copy()

data_spam = df[df['label'] == 'spam'].copy()
show_wordcloud(data_ham, "Ham messages")
show_wordcloud(data_spam, "Spam messages")
from sklearn.model_selection import train_test_split



X = df.loc[:, 'text']

y = df.loc[:, 'label']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import make_pipeline



model = make_pipeline(TfidfVectorizer(), MultinomialNB())



model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score



y_pred = []



for i in X_test:

    y_pred.append(model.predict([i])[0])

    

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')