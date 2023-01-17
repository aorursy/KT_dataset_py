import pandas as pd



df = pd.read_csv('../input/bbc-text.csv')

df.head(10)
# Define the preprocessing steps in a function



import re

from nltk.stem.wordnet import WordNetLemmatizer



stop_words = ['in', 'of', 'at', 'a', 'the']



def pre_process(text):

    

    # lowercase

    text=str(text).lower()

    

    # remove numbers followed by dot (like, "1.", "2.", etc)

    text=re.sub('((\d+)[\.])', '', text)

    

    #remove tags

    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)

    

    # correct some misspellings if there are any you can spot

    text=text.replace('dont', "don't")

    

    # remove special characters except spaces, apostrophes and dots

    text=re.sub(r"[^a-zA-Z0-9.']+", ' ', text)

    

    # remove stopwords

    '''

    Don't include this in the beginning. 

    First check if there are some patterns that may be lost if we remove stopwords.

    '''

    text=[word for word in text.split(' ') if word not in stop_words]

    

    # lemmatize

    lmtzr = WordNetLemmatizer()

    text = ' '.join((lmtzr.lemmatize(i)) for i in text)

    

    return text
# Apply the preprocessing function to each row of text in the dataset

for i in range(len(df)):

    df.text[i] = pre_process(df.text[i])

    

df.head(10)
# Visualize the distribution of categories

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10,6))

df.groupby('category').text.count().plot.bar(ylim=0)

plt.show()
# from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



# Divide the data into 75% training and 25% testing data

train_data = df.text[0:int(0.75*len(df))]

test_data = df.text[int(0.75*len(df))+1:]

train_target = df.category[0:int(0.75*len(df))]

test_target = df.category[int(0.75*len(df))+1:]



# convert the text into numeric form, so that the ML algos can be applied to them

stop_words = ['in', 'of', 'at', 'a', 'the']

ngram_vectorizer = CountVectorizer(binary=True, ngram_range=(1, 3), stop_words=stop_words)

ngram_vectorizer.fit(train_data)

X_train = ngram_vectorizer.transform(train_data)

X_test = ngram_vectorizer.transform(test_data)
# Train the model

model = LogisticRegression() # play around with the parameters in Logisticregression() to find the optimal parameters

model.fit(X_train, train_target)



# make predictions on the test data with the model, and check its accuracy

test_acc = accuracy_score(test_target, model.predict(X_test))

print('Test accuracy: {0:.2f}%'.format(100*test_acc))
import seaborn as sns

from sklearn.metrics import confusion_matrix



conf_mat = confusion_matrix(df.category[int(0.75*len(df))+1:], model.predict(X_test))

fig, ax = plt.subplots(figsize=(10,8))

sns.heatmap(conf_mat, annot=True, fmt='d', xticklabels=df.category.unique(), yticklabels=df.category.unique())

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()
from sklearn import metrics

print(metrics.classification_report(test_target, model.predict(X_test), target_names=df.category.unique()))