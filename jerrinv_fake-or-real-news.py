# initiating gpu using tensorflow.

import tensorflow as tf

from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

config.log_device_placement = True

sess = tf.Session(config=config)

set_session(sess)
import pandas as pd

import numpy as np

import re

from sklearn.feature_extraction.text import CountVectorizer

#import matplotlib as plt

import matplotlib.pyplot as plt
news = pd.read_csv('../input/news_data.csv')
print(news.columns.tolist())
news.loc[390]
news.authors = news.authors.replace("[", "").replace("]","")
news.authors = news.authors.fillna("['No Author']")
news.authors = news.authors.replace("(['No Author'])", "['No Author']")
news.source = news.source.fillna("No Source")
news.text = news.text.fillna("No Text")
def replace_one(row):    

    replace_www = re.sub(r'www.[^ ]+', '', row)

    replace_lower = replace_www.lower()

    result = re.sub(r'[^A-Za-z ]','',replace_lower)

    return result.strip()

news.title = news.title.apply(replace_one)

news.text = news.text.apply(replace_one)
word_grams = CountVectorizer(analyzer = "word", ngram_range = (2,4), stop_words="english")

words = word_grams.fit_transform(news.text)

for c, value in enumerate(word_grams.get_feature_names()):

    news[value] = pd.Series(words[:, c].toarray().ravel())
word_grams = CountVectorizer(analyzer = "word", ngram_range = (2,3), stop_words="english")

words = word_grams.fit_transform(news.title)

for c, value in enumerate(word_grams.get_feature_names()):

    news[value] = pd.Series(words[:, c].toarray().ravel())
news.to_csv ('./news_data_cleaned.csv', index = None, header=True)

news_cleaned = pd.read_csv('../input/fake-or-real-news/news_data_cleaned.csv')
len(news_cleaned.columns)
news_cleaned = news_cleaned.drop(columns=['text','title'])
news_cleaned.head()
news_cleaned.authors = news_cleaned.authors.replace("[]", "['No Author']")
news_cleaned.head()
news_cleaned['authors'] = news_cleaned['authors'].astype("category")

news_cleaned['source'] = news_cleaned['source'].astype("category")

news_cleaned['class'] = news_cleaned['class'].astype("category")

news_cleaned['authors'] = news_cleaned['authors'].cat.codes

news_cleaned['source'] = news_cleaned['source'].cat.codes

news_cleaned['class'] = news_cleaned['class'].cat.codes
news_cleaned.head()
news_train = news_cleaned.drop(columns=['class'])

news_test = news_cleaned['class']
print(news_train.head())

print(news_test.head())
from sklearn.model_selection import train_test_split

from sklearn import tree
x_train,x_test,y_train,y_test = train_test_split(news_train,news_test,test_size=0.3,random_state=10)
Classifier = tree.DecisionTreeClassifier(criterion="entropy")
Classifier = Classifier.fit(x_train,y_train)
y_pred = Classifier.predict(x_test)
max_depths = np.linspace(1, 32, 32, endpoint=True)



from sklearn.metrics import roc_curve, auc

train_results = []

test_results = []

for max_depth in max_depths:

    Classifier = tree.DecisionTreeClassifier(max_depth=5,max_features=10000)

    Classifier.fit(x_train, y_train)



    train_pred = Classifier.predict(x_train)



    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous train results

    train_results.append(roc_auc)



    y_pred = Classifier.predict(x_test)



    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

    roc_auc = auc(false_positive_rate, true_positive_rate)

    # Add auc score to previous test results

    test_results.append(roc_auc)



from matplotlib.legend_handler import HandlerLine2D



line1, = plt.plot(max_depths, train_results, 'b', label="Train AUC")

line2, = plt.plot(max_depths, test_results, 'r', label="Test AUC")



plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})



plt.ylabel('AUC score')

plt.xlabel('Tree depth')

plt.show()
from sklearn.metrics import mean_squared_error, r2_score
print("Mean squared error: %.2f"% mean_squared_error(y_test, y_pred))

print('r2 score: %.2f' % r2_score(y_test, y_pred))
# Model (can also use single decision tree)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=10)



# Train

model.fit(news_train, news_test)

# Extract single tree

estimator = model.estimators_[5]



from sklearn.tree import export_graphviz

# Export as dot file

export_graphviz(estimator, out_file='tree.dot', 

                feature_names = news_train.columns,

                class_names = ['FakeNewsContent','RealNewsContent'],

                rounded = True, proportion = False, 

                precision = 2, filled = True)



# Convert to png using system command (requires Graphviz)

from subprocess import call

call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])



# Display in jupyter notebook

from IPython.display import Image

Image(filename = 'tree.png')