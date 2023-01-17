# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.linear_model import LogisticRegression

from nltk.stem.snowball import SnowballStemmer

from nltk.corpus import stopwords

import re

from sklearn.metrics import accuracy_score









# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_excel("/kaggle/input/Sample data- names and categories - 100k rows.xlsx",nrows=1000)

df = df[df['Product_name'].notnull() ]

df = df[df['Category'].notnull() ]

df = df[df['Sub-Category'].notnull() ]
df.head()
len(df)
df["Category"].nunique()
df["Sub-Category"].nunique()
df["Sub-Category"].value_counts()
df["Category"].value_counts()
df = df[df["Category"]!="TOPICALS"]
df = df[df["Category"]!="OTHER CANNABIS"]
df["Sub-Category"].value_counts()
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



pal = sns.color_palette()



test_qs = pd.Series(df['Product_name'].tolist()).astype(str)



dist_test = test_qs.apply(len)

plt.figure(figsize=(15, 10))

plt.hist(dist_test, bins=200, range=[0, 200], color=pal[1], normed=True, alpha=0.5, label='Product_names')

plt.title('Normalised histogram of character count in product names', fontsize=15)

plt.legend()

plt.xlabel('Number of characters', fontsize=15)

plt.ylabel('Probability', fontsize=15)

dist_test = test_qs.apply(lambda x: len(x.split(' ')))

plt.figure(figsize=(15, 10))

plt.hist(dist_test, bins=200, range=[0, 50], color=pal[1], normed=True, alpha=0.5, label='Product_names')

plt.title('Normalised histogram of Word count in product names', fontsize=15)

plt.legend()

plt.xlabel('Number of Words', fontsize=15)

plt.ylabel('Probability', fontsize=15)
from wordcloud import WordCloud

cloud = WordCloud(width=1440, height=1080).generate(" ".join(test_qs.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
numbers = np.mean(test_qs.apply(lambda x: max([y.isdigit() for y in x])))
print('Questions with numbers: {:.2f}%'.format(numbers * 100))
from collections import Counter



# If a word appears only once, we ignore it completely (likely a typo)

# Epsilon defines a smoothing constant, which makes the effect of extremely rare words smaller

def get_weight(count, eps=10000, min_count=2):

    if count < min_count:

        return 0

    else:

        return 1 / (count + eps)



eps = 5000 

words = (" ".join(test_qs)).lower().split()

counts = Counter(words)

weights = {word: get_weight(count) for word, count in counts.items()}
print('Most common words and weights: \n')

print(sorted(weights.items(), key=lambda x: x[1] if x[1] > 0 else 9999)[:10])

print('\nLeast common words and weights: ')

print(sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10])
df.head()
len(df)
df.to_csv("newdata.csv",index=False)
df['Product_name'] = df.Product_name.str.replace('[^0-9a-zA-Z]', ' ')
df.head()
sw = stopwords.words('english')

def stopwords(text):

    '''a function for removing the stopword'''

    # removing the stop words and lowercasing the selected words

    text = [word.lower() for word in text.split() if word.lower() not in sw]

    # joining the list of words with space separator

    return " ".join(text)



df["Product_name"] = df["Product_name"].apply(stopwords)

df.head()
stemmer = SnowballStemmer("english")



def stemming(text):    

    '''a function which stems each word in the given text'''

    text = [stemmer.stem(word) for word in text.split()]

    return " ".join(text) 
df["Product_name"] = df["Product_name"].apply(stemming)

df.head()
test_qs = pd.Series(df['Product_name'].tolist()).astype(str)

cloud = WordCloud(width=1440, height=1080,collocations=False).generate(" ".join(test_qs.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
y = df["Category"]

X = df.drop(['Category','Sub-Category'],axis=1)
X.head()
y.head()
tfv = TfidfVectorizer(min_df=2,  max_features=None, 

            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',

            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,

            stop_words = 'english')

X = tfv.fit_transform(X['Product_name'])
tfv.get_feature_names()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=25, random_state=0).fit(X)
len(kmeans.labels_)
df = pd.DataFrame (kmeans.labels_)



## save to xlsx file



filepath = 'my_excel_file.xlsx'



df.to_excel(filepath, index=False)
X.toarray()
xtrain, xvalid, ytrain, yvalid = train_test_split(X, y, 

                                                  stratify=y, 

                                                  random_state=42, 

                                                  test_size=0.1, shuffle=True)
print (xtrain.shape)

print (xvalid.shape)

print (ytrain.shape)

print (yvalid.shape)
xtrain.toarray()
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=25, random_state=0).fit(X)
def multiclass_logloss(actual, predicted, eps=1e-15):

    """Multi class version of Logarithmic Loss metric.

    :param actual: Array containing the actual target classes

    :param predicted: Matrix with class predictions, one probability per class

    """

    # Convert 'actual' to a binary array if it's not already:

    if len(actual.shape) == 1:

        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))

        for i, val in enumerate(actual):

            actual2[i, val] = 1

        actual = actual2



    clip = np.clip(predicted, eps, 1 - eps)

    rows = actual.shape[0]

    vsota = np.sum(actual * np.log(clip))

    return -1.0 // rows * vsota
clf = LogisticRegression(C=1.0)

clf.fit(xtrain, ytrain)

predictions = clf.predict_proba(xvalid)

predictions
new_predictions = [0]*len(predictions)

for i in range(len(predictions)):

    new_predictions[i] = predictions[i].argmax()
new_predictions
yvalid
clf.classes_
type(yvalid)
tempvalid = yvalid.replace(to_replace=['ACCESSORIES', 'INGESTIBLES', 'INHALABLES'],value=[0,1,2])
accuracy_score(new_predictions,tempvalid)
from sklearn.metrics import classification_report

print(classification_report(tempvalid, new_predictions, target_names=['ACCESSORIES', 'INGESTIBLES', 'INHALABLES']))
# from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

# clf = RandomForestClassifier(n_estimators=100,oob_score=True,max_features=5)

# clf.fit(xtrain, ytrain)

# predictions = clf.predict_proba(xvalid)

# new_predictions = [0]*len(predictions)

# for i in range(len(predictions)):

#     new_predictions[i] = predictions[i].argmax()
print("Accuracy Score is " + str(accuracy_score(new_predictions,tempvalid)*100))