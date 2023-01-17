import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visuvalization


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data=pd.read_csv('/kaggle/input/sentiment-analysis-on-movie-reviews/train.tsv.zip', sep='\t')
# Any results you write to the current directory are saved as output.
data.head()
data.info()
data.Sentiment.value_counts()

Sentiment_count=data.groupby('Sentiment').count()
plt.bar(Sentiment_count.index.values, Sentiment_count['Phrase'])
plt.xlabel('Review Sentiments')
plt.ylabel('Number of Review')
plt.show()


from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer # Regular Expression
from sklearn.model_selection import train_test_split
from sklearn import tree # Model Generation Using Decision Tree
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

#Re
#tokenizer to remove unwanted elements from out data like symbols and numbers
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(lowercase=True,stop_words='english',ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(data['Phrase'])

X_train, X_test, y_train, y_test = train_test_split(text_counts, data['Sentiment'], test_size=0.3, random_state=1)
# Model Generation Using Decision Tree
clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, predicted))

from sklearn.feature_extraction.text import TfidfVectorizer
##feature extraction
tf=TfidfVectorizer()
text_tf= tf.fit_transform(data['Phrase'])
X_train, X_test, y_train, y_test = train_test_split(text_tf, data['Sentiment'], test_size=0.3, random_state=123)
clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, predicted))