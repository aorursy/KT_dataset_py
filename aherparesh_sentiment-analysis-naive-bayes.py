import math
import random
from collections import defaultdict
from pprint import pprint
from collections import Counter
from nltk.corpus import stopwords
import re
import string
import nltk

# Prevent future/deprecation warnings from showing in output
import warnings
warnings.filterwarnings(action='ignore')

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set global styles for plots
sns.set_style(style='white')
sns.set_context(context='notebook', font_scale=1.3, rc={'figure.figsize': (16,9)})
data=pd.read_csv('../input/hotel-reviews/7282_1.csv')
data.head()
review=pd.DataFrame(data.groupby('reviews.rating').size().sort_values(ascending=False).rename('No of Users').reset_index())
review.head()
import seaborn as sns 
import matplotlib.pyplot as plt
sns.set(style="whitegrid")

f, ax = plt.subplots(figsize=(15, 10))
sns.set_color_codes("pastel") 
sns.barplot(y="reviews.rating", x="No of Users", data=review.iloc[:20, :10],label="Score", color="pink")

ax.legend(ncol=2, loc="upper left", frameon=True)
ax.set(xlabel="No of People",ylabel="Rating") 
sns.despine(left=True, bottom=True)
plt.show()
df=data[['reviews.text','reviews.rating']]
df.head()
df=df.dropna()
df[df['reviews.rating'] != 3]
df['labels'] = np.where(df['reviews.rating'] > 2, 1, 0)
df.head()
df.tail()
stop = set(stopwords.words('english'))
def clean_document(doco):
    punctuation = string.punctuation
    punc_replace = ''.join([' ' for s in punctuation])
    doco_link_clean = re.sub(r'http\S+', '', doco)
    doco_clean_and = re.sub(r'&\S+', '', doco_link_clean)
    doco_clean_at = re.sub(r'@\S+', '', doco_clean_and)
    doco_clean = doco_clean_at.replace('-', ' ')
    doco_alphas = re.sub(r'\W +', ' ', doco_clean)
    trans_table = str.maketrans(punctuation, punc_replace)
    doco_clean = ' '.join([word.translate(trans_table) for word in doco_alphas.split(' ')])
    doco_clean = doco_clean.split(' ')
    p = re.compile(r'\s*\b(?=[a-z\d]*([a-z\d])\1{3}|\d+\b)[a-z\d]+', re.IGNORECASE)
    doco_clean = ([p.sub("", x).strip() for x in doco_clean])
    doco_clean = [word.lower() for word in doco_clean if len(word) > 2]
    doco_clean = ([i for i in doco_clean if i not in stop])
#     doco_clean = [spell(word) for word in doco_clean]
#     p = re.compile(r'\s*\b(?=[a-z\d]*([a-z\d])\1{3}|\d+\b)[a-z\d]+', re.IGNORECASE)
    doco_clean = ([p.sub("", x).strip() for x in doco_clean])
#     doco_clean = ([spell(k) for k in doco_clean])
    return doco_clean
review_clean = [clean_document(doc) for doc in df['reviews.text']]
sentences = [' '.join(r) for r in review_clean]
df['cleantext']=sentences
df.head()
def top_words(data):
        words_list = data.split(' ')
        counts = Counter(words_list)
        top_words = counts.most_common(20)
        length_of_list = len(top_words)
        index = np.arange(length_of_list)
        print(top_words)
        count_values = [x[1] for x in top_words]
        count_words = [x[0] for x in top_words]
        fig = plt.figure(figsize = (16,9))
        bar_width = 0.4
        rects1 = plt.bar(index, count_values)
        plt.xticks(index + bar_width, count_words, rotation=0)
        plt.show()
train_positive_sentiment = df[df.labels == 1]
positive_words = ' '.join(train_positive_sentiment['cleantext'])
print("Top words in Positive Sentiment")
top_words(positive_words)
train_positive_sentiment = df[df.labels == 0]
positive_words = ' '.join(train_positive_sentiment['cleantext'])
print("Top words in Negative Sentiment")
top_words(positive_words)
from sklearn.model_selection import train_test_split

X = df.cleantext
y = df.labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(max_features=1000, binary=True)

X_train_vect = vect.fit_transform(X_train)
counts = df.labels.value_counts()
print(counts)

# print("\nPredicting only -1 = {:.2f}% accuracy".format(counts[-1] / sum(counts) * 100))
from imblearn.over_sampling import SMOTE

sm = SMOTE()

X_train_res, y_train_res = sm.fit_sample(X_train_vect, y_train)
unique, counts = np.unique(y_train_res, return_counts=True)
print(list(zip(unique, counts)))
from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()

nb.fit(X_train_res, y_train_res)

nb.score(X_train_res, y_train_res)
X_test_vect = vect.transform(X_test)

y_pred = nb.predict(X_test_vect)

y_pred
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

print("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))
print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))
print("\nCOnfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(nb.predict(vect.transform(['dirty rooms and beds are not align'])))
# print(nb.predict(vect.transform(['bjoirj ido vhvhhhghghhghhv'])))
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(X_train_res, y_train_res)
ypred=model1.score(X_train_res, y_train_res)
ypred
print(nb.predict(vect.transform(['rooms are very beautiful and nice'])))