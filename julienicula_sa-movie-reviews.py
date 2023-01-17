import numpy as np

import pandas as pd

import string

import matplotlib.pyplot as plot

import seaborn as seaborn

from sklearn.model_selection import train_test_split



train_df = pd.read_csv('../input/train.tsv', sep='\t')



# a glimpse at the training data

train_df.head()
figure = plot.figure(figsize=(10, 5))

seaborn.countplot(data=train_df, x='Sentiment')

plot.show()

def get_count():

    s0 = train_df[train_df.Sentiment == 0].Sentiment.count()

    s1 = train_df[train_df.Sentiment == 1].Sentiment.count()

    s2 = train_df[train_df.Sentiment == 2].Sentiment.count()

    s3 = train_df[train_df.Sentiment == 3].Sentiment.count()

    s4 = train_df[train_df.Sentiment == 4].Sentiment.count()

    return s0, s1, s2, s3, s4



s0, s1, s2, s3, s4 = get_count()

print(s0, s1, s2, s3, s4)



df0 = s2 // s0 - 1

df1 = s2 // s1 - 1

df3 = s2 // s3 - 1

df4 = s2 // s4 - 1

 

train_df = train_df.append([train_df[train_df.Sentiment == 0]] * df0, ignore_index=True)

train_df = train_df.append([train_df[train_df.Sentiment == 1]] * df1, ignore_index=True)

train_df = train_df.append([train_df[train_df.Sentiment == 3]] * df3, ignore_index=True)

train_df = train_df.append([train_df[train_df.Sentiment == 4]] * df4, ignore_index=True)

train_df = train_df.append([train_df[train_df.Sentiment == 0][0 : s2 % s0]], ignore_index=True)

train_df = train_df.append([train_df[train_df.Sentiment == 1][0 : s2 % s1]], ignore_index=True)

train_df = train_df.append([train_df[train_df.Sentiment == 3][0 : s2 % s3]], ignore_index=True)

train_df = train_df.append([train_df[train_df.Sentiment == 4][0 : s2 % s4]], ignore_index=True)



s0, s1, s2, s3, s4 = get_count()

print(s0, s1, s2, s3, s4)

figure = plot.figure(figsize=(5, 2.5))

seaborn.countplot(data=train_df, x='Sentiment')

plot.show()
from wordcloud import WordCloud

from nltk.corpus import stopwords

sentiments = [0, 1, 2, 3, 4]

cloud = WordCloud(background_color="white", max_words=20, stopwords=stopwords.words('english'))



def draw_word_clouds(dataframe):

    for i in sentiments: 

        category = cloud.generate(dataframe.loc[dataframe['Sentiment'] == i, 'Phrase'].str.cat(sep='\n'))

        plot.figure(figsize=(5, 2.5))

        plot.imshow(category)

        plot.axis("off")

        plot.title(i)

        plot.show()



draw_word_clouds(train_df)
import re

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from joblib import Parallel, delayed

import string 

import time 



lemmatizer = WordNetLemmatizer() 

stop_words  = stopwords.words('english')

stop_words.extend(['movie', 'film', 'series', 'story', 'one', 'like'])



def clean_review(review):

#     review = re.sub("[^a-zA-Z]", " ", review)

    tokens = review.lower().split()

    filtered_tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]

    return " ".join(filtered_tokens)



start = time.time()

clean_train_data = train_df.copy()

clean_train_data['Phrase'] = Parallel(n_jobs=4)(delayed(clean_review)(review) for review in train_df['Phrase'])

end = time.time()

print("Cleaning Training Data - Processing time = ", end - start)



# remove missing values

print("Clean entries: ", clean_train_data.shape[0], " out of ", train_df.shape[0])
target = clean_train_data.Sentiment

train_X_, validation_X_, train_y, validation_y = train_test_split(clean_train_data['Phrase'], target, test_size=0.25, random_state=21)

from sklearn.feature_extraction.text import TfidfVectorizer as tfidf



tfidf_vec = tfidf(min_df=3,  max_features=None, 

        ngram_range=(1, 2), use_idf=1)

start = time.time()

train_X = tfidf_vec.fit_transform(train_X_)

end = time.time()

print("TFIDF finished in: ", end - start)
from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics



model = MultinomialNB()

model.fit(train_X, train_y)

validation_X = tfidf_vec.transform(validation_X_)

predicted = model.predict(validation_X)

expected = validation_y

print(metrics.classification_report(expected, predicted))

print(metrics.accuracy_score(expected, predicted))
test_df = pd.read_csv('../input/test.tsv', sep='\t')



clean_test_data = test_df.copy()

clean_test_data['Phrase'] = Parallel(n_jobs=4)(delayed(clean_review)(review) for review in test_df['Phrase'])

end = time.time()

print("Cleaning Testing Data - Processing time = ", end - start)



# remove missing values

print("Clean entries: ", clean_test_data.shape[0], " out of ", test_df.shape[0])

test_X = tfidf_vec.transform(clean_test_data['Phrase'])



test_predictions = model.predict(test_X)



output = pd.DataFrame({

    'PhraseId': test_df['PhraseId'],

    'Sentiment': test_predictions

})



output.to_csv('submission.csv', index=False)