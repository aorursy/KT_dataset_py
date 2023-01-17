import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

import nltk

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import wordcloud

from sklearn import metrics

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



sms_data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding = 'latin-1')
# Let's check the head of our dataframe

sms_data.head()
sms_data.drop(sms_data.iloc[:,2:], axis = 1, inplace = True)
# Let's check our data again

sms_data.head()
sms_data.rename(columns = {'v1': 'label', 'v2' : 'sms'}, inplace = True)
sms_data.head()
# Let's check whether ther are null values present in the data or not

sms_data.isnull().any()
sms_data.label.value_counts().plot.bar(rot = 0)

plt.xlabel('Class')

plt.ylabel('Frequency')

plt.title('SMS class distribution')
sms_data['spam'] = pd.get_dummies(sms_data['label'], drop_first = True)



sms_data.head()
data_ham = sms_data[sms_data['spam'] == 0]

data_spam = sms_data[sms_data['spam'] == 1]
def show_wordcloud(data_spam_or_ham, title):

    text = ' '.join(data_spam_or_ham['sms'].astype(str).tolist())

    stopwords = set(wordcloud.STOPWORDS)

    

    fig_wordcloud = wordcloud.WordCloud(stopwords = stopwords,background_color = 'lightgrey',

                    colormap='Accent', width = 800, height = 600).generate(text)

    

    plt.figure(figsize = (10,7), frameon = True)

    plt.imshow(fig_wordcloud)  

    plt.axis('off')

    plt.title(title, fontsize = 20 )

    plt.show()
show_wordcloud(data_spam, 'Spam SMS')
show_wordcloud(data_ham, 'Ham SMS')
X = sms_data['sms']

y = sms_data['spam']
def process_data(message):

    ps = PorterStemmer()   # Porter Stemmer Object



    corpus = []



    for i in range(0, len(message)):

        review = re.sub('[^A-Za-z]', ' ', message[i])

        review = review.lower()

        review = review.split()

    

        review = [ps.stem(word) for word in review if word not in(stopwords.words('english'))]

        review = ' '.join(review)

        corpus.append(review)

    return corpus
corpus = process_data(X)
corpus
tfidf = TfidfVectorizer(max_features = 4000)

X = tfidf.fit_transform(corpus).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 

                                                    random_state = 101, shuffle = True)
print(X_train.shape)

print(X_test.shape)
rf_model = RandomForestClassifier(random_state = 101)



rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print(metrics.classification_report(y_test, y_pred))



confusion_matrix = metrics.confusion_matrix(y_test,y_pred)

display(pd.DataFrame(data = confusion_matrix, columns = ['Predicted 0', 'Predicted 1'],

            index = ['Actual 0', 'Actual 1']))