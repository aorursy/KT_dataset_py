# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from math import log, sqrt
%matplotlib inline


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
spam_path = "../input/spam-text-message-classification/SPAM text message 20170820 - Data.csv"
spam_data = pd.read_csv(spam_path)
spam_data
spam_data.isna().any()
from sklearn.preprocessing import LabelEncoder
label_cat = spam_data['Category'].copy()
le = LabelEncoder()
label_cat = le.fit_transform(label_cat)
spam_data['Label Category'] = label_cat
spam_data.drop(['Category'], axis=1, inplace=True)
spam_data.head()

spam_rows = spam_data.loc[spam_data['Label Category'] == 1].copy()
ham_rows = spam_data.loc[spam_data['Label Category'] == 0].copy()
ham_rows.head()
print(ham_rows.shape)
print(spam_rows.shape)
spam_words = ' '.join(list(spam_data[spam_data['Label Category'] == 1]['Message']))
spam_wc = WordCloud(width = 512, height = 512).generate(spam_words)
plt.figure(figsize=(10,10))
plt.imshow(spam_wc)
plt.axis('off')
plt.show()
ham_words = ' '.join(list(spam_data[spam_data['Label Category'] == 0]['Message']))
ham_wc = WordCloud(width = 512, height = 512).generate(ham_words)
plt.figure(figsize=(10,10))
plt.imshow(ham_wc)
plt.axis('off')
plt.show()
def process_mes(message, lower_case=True, stem=True, stop_words=True, gram=1):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 2]
    words = [w for w in words if w.isalpha()]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i+gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words
b = []
for mes in spam_data['Message']:
    b.append(' '.join(process_mes(mes, gram=1)))
spam_data['Processed Message'] = b
spam_data.head()
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
X = spam_data['Processed Message'].copy()
y = spam_data['Label Category'].copy()
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
cv = CountVectorizer()
features = cv.fit_transform(X_train)
#model
#model = GaussianNB()
#gnb.fit(features, y_train)
tuned_parameters = {'kernel':['linear','rbf'], 'gamma':[1e-3,1e-4], 'C':[1, 10, 100, 1000]}
model = GridSearchCV(svm.SVC(), tuned_parameters)
model.fit(features, y_train)
features_valid = cv.transform(X_valid)
preds = model.predict(features_valid)
print(model.best_params_)
print("Accuracy = %f" % ( (y_valid == preds).sum()/ X_valid.shape[0] ))