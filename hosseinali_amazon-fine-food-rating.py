import os

print(os.listdir("../input/amazon-fine-food-reviews"))
import sqlite3

import pandas as pd

con = sqlite3.connect('../input/amazon-fine-food-reviews/database.sqlite')

data = pd.read_sql_query("""select * from Reviews""",con)
import matplotlib.pyplot as plt

data['Score'] = data['Score'].map(lambda x: 0 if x<4 else (1))

data = data.drop_duplicates(subset={"ProductId","UserId","Score","Text"},keep="first",inplace=False)

X = data['Summary']

Y = data['Score']

Y.value_counts().plot(kind='bar',colormap='Paired')

plt.ylabel('Count')

plt.show()
import re

from collections import Counter

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))



X = X.apply( lambda x: x.lower() )

X = X.apply( lambda x: re.sub( re.compile('<.*?>') , ' ' , x ) )

X = X.apply( lambda x: re.sub( r'[?|!|\'|"|#]' , r'' , x ) )

X = X.apply( lambda x: re.sub( r'[.|,|)|(|\|/]' , r'' , x ) )

X = X.apply( lambda x: [i for i in x.split() if i not in stop ] )

X = X.apply( lambda x: Counter(x) )
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
import operator

def most_common_pos(X,epoch):

    total = {}

    ini = 0

    while ini + epoch < len(X):

        most_common = X_train[ini:ini+epoch].sum().most_common(30)

        dic = { w:c for w,c in most_common }

        total = { k: dic.get(k, 0) + total.get(k, 0) for k in set(dic) | set(total) }

        ini += epoch

    sorted_total = sorted(total.items(), key=operator.itemgetter(1))

    sorted_total = sorted_total[::-1]

    sorted_total = sorted_total[:23]

    return sorted_total
X_train_pos = X_train[ Y_train == 1 ]

most_common = most_common_pos( X_train_pos , 300 )
from wordcloud import WordCloud, STOPWORDS



wordcloud = WordCloud( background_color='white',

                      max_words=200,

                      max_font_size=40,

                      random_state=42).generate(str(X_train_pos))



plt.figure(figsize=(8, 4), dpi=90, facecolor='w', edgecolor='k')

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
import numpy as np

vocabulary = {word: index + 1 for index, (word, count) in enumerate(most_common)}

X_train = X_train.apply(lambda x : np.asarray( [vocabulary.get(word, 0) for word, count in x.items()] ))

X_test = X_test.apply(lambda x : np.asarray( [vocabulary.get(word, 0) for word, count in x.items()] ))
from scipy.sparse import csr_matrix

def transform(X):

    rows = []

    cols = []

    data = []

    for row, listy in enumerate(X):

        for col , numy in enumerate(listy):

            rows.append(row)

            cols.append(col)

            data.append(numy)

    return csr_matrix((data, (rows, cols)), shape=(len(X), 20 ))
X_train = transform(X_train)

X_test = transform(X_test)
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression



log_clf = LogisticRegression(solver="liblinear", random_state=42)

scores = cross_val_score(log_clf, X_train , Y_train , cv=5, verbose=3)
print('Scores for all folds: ', str(np.around( scores , 3)))

print('Average Score: ', str(np.around( scores.mean() , 3)))

print('Standard deviation of Scores: ', str(np.around( scores.std() , 4)))
from sklearn.model_selection import cross_val_predict

Y_scores = cross_val_predict(log_clf, X_train , Y_train, cv=3, method="decision_function")
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(Y_train, Y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):

    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)

    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)

    plt.xlabel("Threshold", fontsize=16)

    plt.legend(loc="lower left", fontsize=16)

    plt.ylim([0, 1])



plt.figure(figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)

plt.show()