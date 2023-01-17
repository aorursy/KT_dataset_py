# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('seaborn-whitegrid')



import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score

from sklearn.pipeline import Pipeline

from sklearn.multioutput import MultiOutputClassifier

from sklearn.svm import LinearSVC

from sklearn.linear_model import LogisticRegression
train = pd.read_csv('../input/topic-modeling-for-research-articles-20/Train.csv')

test = pd.read_csv('../input/topic-modeling-for-research-articles-20/Test.csv')

ss = pd.read_csv('../input/topic-modeling-for-research-articles-20/SampleSubmission.csv')

tags = pd.read_csv('../input/topic-modeling-for-research-articles-20/Tags.csv')
train.head()
test.head()
tags.head()
ss.head()
train.info()
train.nunique()
TARGET_COLS = ['Analysis of PDEs', 'Applications',

               'Artificial Intelligence', 'Astrophysics of Galaxies',

               'Computation and Language', 'Computer Vision and Pattern Recognition',

               'Cosmology and Nongalactic Astrophysics',

               'Data Structures and Algorithms', 'Differential Geometry',

               'Earth and Planetary Astrophysics', 'Fluid Dynamics',

               'Information Theory', 'Instrumentation and Methods for Astrophysics',

               'Machine Learning', 'Materials Science', 'Methodology', 'Number Theory',

               'Optimization and Control', 'Representation Theory', 'Robotics',

               'Social and Information Networks', 'Statistics Theory',

               'Strongly Correlated Electrons', 'Superconductivity',

               'Systems and Control']
100 * (train[TARGET_COLS].sum()/(train.shape[0])).sort_values(ascending=False)
from wordcloud import WordCloud, STOPWORDS

wc = WordCloud(stopwords = set(list(STOPWORDS) + ['inside']), random_state = 42)
#Check for top words for a given sub-topic

fig, axes = plt.subplots(2, 2, figsize=(20, 12))

axes = [ax for axes_row in axes for ax in axes_row]

for i, sub_topic_name in enumerate(['Machine Learning', 'Artificial Intelligence', 'Computer Vision and Pattern Recognition', 'Robotics']):

  sub_topic = train[train[sub_topic_name] == 1]

  op = wc.generate(str(sub_topic['ABSTRACT']))

  _ = axes[i].imshow(op)

  _ = axes[i].set_title(sub_topic_name.upper(), fontsize=20)

  _ = axes[i].axis('off')

_ = plt.suptitle('TOP WORDS FOR A GIVEN SUB-TOPIC', fontsize=30)
TOPIC_COLS = ['Computer Science', 'Mathematics', 'Physics', 'Statistics']
#Check for top words for a given topic

fig, axes = plt.subplots(2, 2, figsize=(20, 12))

axes = [ax for axes_row in axes for ax in axes_row]

for i, topic_name in enumerate(TOPIC_COLS):

  topic = train[train[topic_name] == 1]

  op = wc.generate(str(topic['ABSTRACT']))

  _ = axes[i].imshow(op)

  _ = axes[i].set_title(topic_name.upper(), fontsize=20)

  _ = axes[i].axis('off')

_ = plt.suptitle('TOP WORDS FOR A GIVEN TOPIC', fontsize=30)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.feature_extraction import text

from collections import Counter
cv = CountVectorizer()

data_cv = cv.fit_transform(tags['Tags'])

data_dtm = pd.DataFrame(data_cv.toarray(), columns = cv.get_feature_names())

data_dtm.index = tags.index

data_dtm = data_dtm.transpose()

data_dtm.head()
# Find the top 30 words on each category



top_dict = {}

for c in data_dtm.columns:

    top = data_dtm[c].sort_values(ascending = False).head(30)

    top_dict[c]= list(zip(top.index, top.values))



top_dict
for category, top_words in top_dict.items():

    print(category, ":")

    print(', '.join([word for word, count in top_words[0:14]]))

    print('-----------------------------------------------------------------------------------------------------------------------')
# Let's first pull out the top words for each category



words = []

for category in data_dtm.columns:

    top = [word for (word, count) in top_dict[category]]

    for t in top:

        words.append(t)

        

words

add_stop_words = [word for word, count in Counter(words).most_common() if count > 2]





# Add new stop words



stop_words = text.ENGLISH_STOP_WORDS.union(add_stop_words)



vectorizer = TfidfVectorizer(strip_accents = 'unicode', analyzer = 'word', ngram_range = (1, 2), norm = 'l2', max_features = 10000, use_idf = True, stop_words = stop_words)

train_data, eval_data = train_test_split(train, test_size=0.2, random_state=42,shuffle = True)
vectorizer.fit(train_data['ABSTRACT'])

vectorizer.fit(eval_data['ABSTRACT'])



trn_abs = vectorizer.transform(train_data['ABSTRACT'])

val_abs = vectorizer.transform(eval_data['ABSTRACT'])

tst_abs = vectorizer.transform(test['ABSTRACT'])
train_data[TARGET_COLS]
from skmultilearn.problem_transform import BinaryRelevance, LabelPowerset
lp_classifier = LabelPowerset(LogisticRegression(max_iter = 50, verbose = 10))

lp_classifier.fit(trn_abs, train_data[TARGET_COLS])

lp_predictions = lp_classifier.predict(val_abs)
print("Accuracy = ", accuracy_score(eval_data[TARGET_COLS], lp_predictions))

print("F1 score = ", f1_score(eval_data[TARGET_COLS], lp_predictions, average = "micro"))
pipe = Pipeline([('TFidf', TfidfVectorizer(ngram_range = (1,2), stop_words = stop_words, smooth_idf = True)), 

                 ("multilabel", MultiOutputClassifier(LinearSVC( random_state = 42, class_weight = 'balanced')))])
y_train = train_data[TARGET_COLS]
pipe.fit(train_data['ABSTRACT'], y_train)
pipe_pred= pipe.predict(eval_data['ABSTRACT'])
print("Accuracy = ", accuracy_score(eval_data[TARGET_COLS], pipe_pred))

print("F1 score = ", f1_score(eval_data[TARGET_COLS], pipe_pred, average = "micro"))
preds_test = pipe.predict(test['ABSTRACT'])
 ## 1. Setting the target column with our obtained predictions

ss[TARGET_COLS] = preds_test



  ## 2. Saving our predictions to a csv file



ss.to_csv('Submission.csv', index = False)

ss.head()