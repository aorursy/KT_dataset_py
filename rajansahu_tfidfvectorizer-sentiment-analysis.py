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
df=pd.read_csv("/kaggle/input/imdb-sentiments/train.csv")
df.head()
pos_review=0
neg_review=0
for i in range(25000):
    if df["sentiment"][i]==1:
        pos_review+=1
    else:
        neg_review+=1
print(neg_review)
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

objects = ('Postive_Reviews','Negative_Reviews')
y_pos = np.arange(len(objects))
performance = [pos_review,neg_review]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of reviews')
plt.title('IMDB reviews')

plt.show()
import re
corpus = []
for i in range(0, 25000):
    review = re.sub(r'\W', ' ', str(df["text"][i]))
    review = review.lower()
    review = re.sub(r'^br$', ' ', review)
    review = re.sub(r'\s+br\s+',' ',review)
    review = re.sub(r'\s+[a-z]\s+', ' ',review)
    review = re.sub(r'^b\s+', '', review)
    review = re.sub(r'\s+', ' ', review)
    corpus.append(review)
corpus[1]
# Creating the Tf-Idf model directly
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features = 2000, min_df = 3, max_df = 0.6, stop_words = stopwords.words('english'))
X = vectorizer.fit_transform(corpus).toarray()
from sklearn import model_selection, preprocessing
train_x, test_x, train_y, test_y = model_selection.train_test_split(X, df['sentiment'])

# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
#print(train_y)
test_y = encoder.fit_transform(test_y)
#print(valid_y)
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(train_x,train_y)
import matplotlib.pyplot as plt 
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(classifier,test_x, test_y)  # doctest: +SKIP
plt.show()
from sklearn import model_selection, preprocessing, naive_bayes, metrics
from sklearn import decomposition, ensemble
classifier2= naive_bayes.MultinomialNB()
classifier2.fit(train_x,train_y)
import matplotlib.pyplot as plt 
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(classifier2,test_x, test_y)  
plt.show()
classifier3=ensemble.RandomForestClassifier()
classifier3.fit(train_x,train_y)

import matplotlib.pyplot as plt 
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(classifier3,test_x, test_y)  
plt.show()