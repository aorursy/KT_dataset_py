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
from sklearn import preprocessing,metrics

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer,CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import stopwords

from sklearn.metrics import confusion_matrix, roc_curve

import pandas as pd, numpy as np

import re,nltk, string

from string import punctuation

import nltk

nltk.download("popular")
data = pd.read_csv('/kaggle/input/eopinionscom-product-reviews/Eopinions.csv')

df = pd.DataFrame(data)

df.head()
#as a task i am Label Encoding the class

from sklearn.preprocessing import LabelEncoder

le= LabelEncoder()

df['class']= le.fit_transform(df['class'])
df['class'].value_counts()
import seaborn as sns

sns.countplot(df['class'])

#again as a part of task plotting on basis of encoded column
#Pre processing the text

df['new_text'] = df['text'].replace(to_replace=r'[^a-zA-Z ]+',value='', regex=True)

df['new_text'] = df['new_text'].str.replace('((www\.[\s]+)|(https?://[^\s]+))','\0',regex=True)

#df['new_text'] = df['new_text'].str.lower()

#df['new_text'] = df['new_text'].str.split()

df['new_text']
#stop = stopwords.words('english')

#df['new_text']=df['new_text'].apply(lambda x: [item for item in x if item not in stop])

#df['new_text']
#from nltk.tokenize import word_tokenize, sent_tokenize

#def rejoin_words(row):

#    my_list = row['new_text']

#    joined_words = ( " ".join(my_list))

#    return joined_words

#df['new_text'] = df.apply(rejoin_words, axis=1)

#df['new_text']
#w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

#lemmatizer = nltk.stem.WordNetLemmatizer()

#def lemmatize_text(text):

#    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

#df['new_text'] = df['new_text'].apply(lemmatize_text)

#def rejoin_words(row):

#    my_list = row['new_text']

#    joined_words = ( " ".join(my_list))

#    return joined_words

#df['new_text'] = df.apply(rejoin_words, axis=1)

#df['new_text']
from sklearn.feature_extraction.text import CountVectorizer

count_vect=CountVectorizer(max_df=0.95,min_df=2,max_features=100,ngram_range = (1,2),stop_words='english')





dtm=count_vect.fit_transform(df['new_text'])



repr(dtm)



#print(dtm)
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

#from sklearn.feature_extraction.text import TfidfVectorizer

#tf_vect=TfidfVectorizer(min_df=7,max_df=0.3,ngram_range = (1,2))

#tf_matrix=tf_vect.fit_transform(df['new_text'])



cv = count_vect.fit_transform(df['new_text'])



X=cv.toarray().tolist()

y=df.new_text.tolist()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 42)
clf = RandomForestClassifier(n_estimators = 100)

clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


from sklearn.metrics import classification_report

report = confusion_matrix(y_test, y_pred)

report
false_positive, true_positive,threshold = roc_curve(y_test, y_pred)
import matplotlib.pyplot as plt

plt.plot(false_positive, true_positive)