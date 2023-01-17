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
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
train=pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/train.csv')
test=pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/test.csv')
test_original=pd.read_csv('/kaggle/input/janatahack-independence-day-2020-ml-hackathon/test.csv')
text_train=train['TITLE']+train['ABSTRACT']
text_test=test['TITLE']+test['ABSTRACT']
text=pd.concat([text_train,text_test])
# Replace numbers with 'numbr'
processed_text =text.str.replace(r'\d+(\.\d+)?', 'numbr')

# Remove punctuation
processed_text = processed_text.str.replace(r'[^\w\d\s]', ' ')

# Replace whitespace between terms with a single space
processed_text = processed_text.str.replace(r'\s+', ' ')

# Remove leading and trailing whitespace
processed_text = processed_text.str.replace(r'^\s+|\s+?$', '')

# change words to lower case - Hello, HELLO, hello are all the same word
processed_text = processed_text.str.lower()
from nltk.corpus import stopwords

# remove stop words from text messages

stop_words = set(stopwords.words('english'))

processed_text = processed_text.apply(lambda x: ' '.join(
    term for term in x.split() if term not in stop_words))
# Modify words with Word Net Lemmatizer
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

processed_text=processed_text.apply(lambda x: ' '.join(
    wordnet_lemmatizer.lemmatize(term) for term in x.split()))
# Remove word stems using a Porter stemmer
ps = nltk.PorterStemmer()

processed_text = processed_text.apply(lambda x: ' '.join(
    ps.stem(term) for term in x.split()))
processed_text_train=processed_text[:20972]
processed_text_test=processed_text[20972:]
total_size=train.shape[0]
train_size=int(0.8*total_size)

X_train=processed_text_train.head(train_size)
X_test=processed_text_train.tail(total_size-train_size)

y = train.iloc[:, 3:]
y_train=y[0:train_size]
y_test=y[train_size:total_size]
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(min_df=1, smooth_idf=True, norm="l2",
                          tokenizer=lambda x: x.split(),sublinear_tf=True, ngram_range=(1,3))

X_train_multilabel=vectorizer.fit_transform(X_train)
X_test_multilabel=vectorizer.transform(X_test)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

classifier=OneVsRestClassifier(LinearSVC(penalty="l2",loss='hinge'), n_jobs=-1)
classifier.fit(X_train_multilabel,y_train)
predictions=classifier.predict(X_test_multilabel)

print(f1_score(y_test,predictions, average='micro'))
test_multilabel=vectorizer.transform(processed_text_test)
predictions=classifier.predict(test_multilabel)
submission=pd.DataFrame(predictions, columns=['Computer Science','Physics','Mathematics','Statistics','Quantitative Biology','Quantitative Finance'])
submission=pd.concat([test_original['ID'],submission],axis=1)
submission.to_csv("topic.csv", index=False)
print(submission.shape)
submission.head()
