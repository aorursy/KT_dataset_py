import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import nltk

from nltk import FreqDist

from nltk.corpus import stopwords

import re



from sklearn.preprocessing import MultiLabelBinarizer



# Load the data

df = pd.read_csv('../input/videogamesales/vgsales.csv')

# Reduce dataframe to feature 'Name', and label 'Genre'

df = df[['Name','Genre']].copy()

df.head()
tokens = nltk.word_tokenize(' '.join(df.Name.values))

text = nltk.Text(tokens)

fdist = FreqDist(text)

print(fdist)

fdist.most_common(50)
fdist.plot(50,cumulative=True,title='Top 50 Word Frequency Cumulative Plot')
text.collocations()
genres = list(df.Genre.unique())

d = {}

for g in genres:

    names = list(df[df.Genre == g].Name.values)

    tokens = nltk.word_tokenize(' '.join(names))

    types = set(tokens)

    lexical_diversity = round(len(types) / len(tokens),3)

    d[g] = (len(tokens), len(types), lexical_diversity)

    

    #print(f"{g}: TOKENS: {len(tokens)}, TYPES: {len(types)}, LEXICAL DIVERSITY: {lexical_diversity}")

table = pd.DataFrame.from_dict(d,orient='index',columns=['tokens','type','lexical_diversity'])

display(table.sort_values('lexical_diversity'))
def clean_text(text):

    text = text.lower()

    tokens = nltk.word_tokenize(text)

    tokens = [t for t in tokens if t.isalpha()]

    tokens = [t for t in tokens if t not in stopwords.words('english')]

    # Remove roman numerals using regex

    roman_re = r'\bM{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})\b'

    tokens = [t for t in tokens if not re.match(roman_re,t,flags=re.IGNORECASE).group()]

    

    text = ' '.join(tokens).strip()

    

    return text



df.Name = df.Name.apply(lambda n: clean_text(n))

df.sample(20)
df.Genre.value_counts(ascending=True).plot(kind='barh')

plt.show()
genres = list(df.Genre.unique())

d = {}

for g in genres:

    names = list(df[df.Genre == g].Name.values)

    tokens = nltk.word_tokenize(' '.join(names))

    types = set(tokens)

    lexical_diversity = round(len(types) / len(tokens),3)

    d[g] = (len(tokens), len(types), lexical_diversity)

    

    #print(f"{g}: TOKENS: {len(tokens)}, TYPES: {len(types)}, LEXICAL DIVERSITY: {lexical_diversity}")

table = pd.DataFrame.from_dict(d,orient='index',columns=['tokens','type','lexical_diversity'])

display(table.sort_values('lexical_diversity'))
tokens = nltk.word_tokenize(' '.join(df.Name.values))

text = nltk.Text(tokens)

fdist = FreqDist(text)

print(fdist)

fdist.most_common(50)
fdist.plot(50,cumulative=True,title='Top 50 Word Frequency Cumulative Plot')
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split



tfidf_vectorizer = TfidfVectorizer(analyzer='word',ngram_range=(1,2))



# split dataset into training and validation set

y = df.Genre

x = df.Name

xtrain, xval, ytrain, yval = train_test_split(x,y, test_size = 0.2)



# create the TF-IDF features

xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)

xval_tfidf = tfidf_vectorizer.transform(xval)



from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression



from sklearn.metrics import f1_score

from sklearn import metrics



mnb = MultinomialNB()

lr = LogisticRegression(max_iter=1000)



# fit model on train data

mnb.fit(xtrain_tfidf, ytrain)

lr.fit(xtrain_tfidf, ytrain)



# make predictions for validation set

mnb_pred = mnb.predict(xval_tfidf)

lr_pred = lr.predict(xval_tfidf)



# evaluate performance

mnb_acc = metrics.accuracy_score(yval, mnb_pred)

mnb_acc = round(mnb_acc,2)

lr_acc = metrics.accuracy_score(yval, lr_pred)

lr_acc = round(lr_acc,2)
print(f"Accuracy Scores:\nMultinomial Naive Bayes: {mnb_acc}")

print(f"Logistic Regression: {lr_acc}")
pred_df = pd.DataFrame(xval)

pred_df['actual'] = yval

pred_df['prediction'] = lr_pred

pred_df.sample(10)