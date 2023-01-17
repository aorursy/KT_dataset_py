import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/train.csv')

test = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/test.csv')

sample_submission = pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/sample_submission_UVKGLZE.csv')

df
sample_submission
columns = df.columns

columns
df['No. of topics'] = (df['Computer Science'] + df['Physics'] + df['Mathematics'] + 

                       df['Statistics'] + df['Quantitative Biology'] + df['Quantitative Finance'])



df.head(3)
df['No. of topics'].value_counts()
from collections import Counter

results = Counter()

df['ABSTRACT'].str.lower().str.split().apply(results.update)
one_timers = []



for k,v in results.items():

    if v == 1:

        one_timers.append(k)
from wordcloud import WordCloud, STOPWORDS

new_stopwords = ['based', 'paper', 'we', 'the', 'model', 'using', 'show', 'that' 'used', 

                 'use', '!', '$', '%', '&', ',', '.', 'we', 'method', 'problem', 'models']

STOPWORDS.update(new_stopwords)

STOPWORDS.update(one_timers)
def remove_stopwords(text):

    from nltk.tokenize import word_tokenize



    text_tokens = word_tokenize(text)



    tokens_without_sw = [word for word in text_tokens if not word in STOPWORDS]

    

    filtered_sentence = (" ").join(tokens_without_sw)



    return filtered_sentence
def data_clean(df):

    df['text'] = df['TITLE'] + df['ABSTRACT']

    df['text'] = df['text'].apply(remove_stopwords)

    

    return df
cleaned_data = data_clean(df)
from collections import Counter

results = Counter()

cleaned_data['text'].str.lower().str.split().apply(results.update)

counter_df = pd.DataFrame.from_dict(results, orient='index')

counter_df['Total'] = counter_df[0]

counter_df
labels = ['Computer Science', 'Physics', 'Mathematics','Statistics', 

          'Quantitative Biology', 'Quantitative Finance']







for label in labels:

    from collections import Counter

    results = Counter()

    cleaned_data[cleaned_data[label]==1]['text'].str.lower().str.split().apply(results.update)

    temp_counter_df = pd.DataFrame.from_dict(results, orient='index')

    temp_counter_df[label] = temp_counter_df[0]

    counter_df = counter_df.merge(how='outer', left_index=True, right_index=True, right=temp_counter_df[label])



    

counter_df.sort_values(by='Total', axis=0, ascending=False).head(15)
counter_df
import matplotlib.pyplot as plt

word_string=" ".join(cleaned_data['text'].str.lower())

wordcloud = WordCloud(stopwords=STOPWORDS).generate(word_string)



plt.subplots(figsize=(15,15))

plt.clf()

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
labels = ['Computer Science', 'Physics', 'Mathematics','Statistics', 

          'Quantitative Biology', 'Quantitative Finance']



for label in labels:

    print(label)

    word_string=" ".join(cleaned_data[cleaned_data[label]==1]['text'].str.lower())

    wordcloud = WordCloud(stopwords=STOPWORDS).generate(word_string)



    



    plt.subplots(figsize=(15,15))

    plt.title(label)

    plt.clf()

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()



df.head(3)
X = df[['ID','TITLE', 'ABSTRACT']]

y = df[['Computer Science', 'Physics', 'Mathematics','Statistics', 

          'Quantitative Biology', 'Quantitative Finance']] 



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.33)



submission = pd.DataFrame(X_test['ID'])



X_train = data_clean(X_train)['text']

X_test = data_clean(X_test)['text']



from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn import metrics



labels = ['Computer Science', 'Physics', 'Mathematics','Statistics', 

          'Quantitative Biology', 'Quantitative Finance']





for label in labels:

    

    print(y_test[label].value_counts())

    

    text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),

                         ('clf', LinearSVC(random_state=0)),

    ])



    text_clf.fit(X_train, y_train[label])  



    predictions = text_clf.predict(X_test)



    submission[label] = predictions



    print('')

    print(metrics.confusion_matrix(y_test[label],predictions))

    print('')

    print(metrics.classification_report(y_test[label],predictions))

    print('')

    print('')

    print('')

    print('')
submission
test
X = df[['ID','TITLE', 'ABSTRACT']]

y = df[['Computer Science', 'Physics', 'Mathematics','Statistics', 

          'Quantitative Biology', 'Quantitative Finance']] 





submission = pd.DataFrame(test['ID'])

#submission = test



X = data_clean(X)['text']

test = data_clean(test)['text']





from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

from sklearn import metrics



labels = ['Computer Science', 'Physics', 'Mathematics','Statistics', 

          'Quantitative Biology', 'Quantitative Finance']



for label in labels:

    

    text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),

                         ('clf', LinearSVC(random_state=0)),

    ])



    text_clf.fit(X, y[label])  



    predictions = text_clf.predict(test)



    submission[label] = predictions
submission
filename = 'submission.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)