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
import nltk

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from tqdm import tqdm

import seaborn as sns

import re



import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



nltk.download('stopwords')

nltk.download('wordnet')
train_path = '../input/nlp-getting-started/train.csv'

test_path = '../input/nlp-getting-started/test.csv'

submission_path = '../input/nlp-getting-started/sample_submission.csv'



train = pd.read_csv(train_path)

test = pd.read_csv(test_path)

submission = pd.read_csv(submission_path)
train.head()
train.isnull().sum()
texts = train['text'].to_list()

labels = train['target'].to_list()
lemmatizer = WordNetLemmatizer()

stop_words = stopwords.words('english')
def preprocess(texts):

    x_train = []

    for sent in tqdm(texts):

        sent = re.sub("[^a-zA-Z]", " ", sent)

        sent = sent.lower().split()

        sent = [lemmatizer.lemmatize(word) for word in sent if word not in set(stop_words)]

        sent = ' '.join(sent)

        x_train.append(sent)

    return x_train
X = preprocess(texts)
tfidf = TfidfVectorizer(max_features=5000, analyzer='word', ngram_range=(1,2), stop_words='english')

X_train = tfidf.fit_transform(X).toarray()
x_train, x_test, y_train, y_test = train_test_split(X_train, labels, test_size=0.2, random_state=0)
classifiers = [RandomForestClassifier(n_estimators=100, random_state=0), MultinomialNB()]



for clf in classifiers:

    

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    print('Classifier: {} \tAccuracy score: {}%'.format(clf.__class__.__name__, 

                                                        accuracy_score(y_pred, y_test)))

    

    print('Classification report:')

    print(classification_report(y_test, y_pred))

    

    print('Confusion matrix:')

    

    conf_mat = confusion_matrix(y_test, y_pred)

    df = pd.DataFrame(conf_mat, columns=['Good', 'Bad'])

    plt.figure(figsize=(20, 20))

    sns.heatmap(df, annot=True)

    plt.show()
test_text = test['text'].to_list()

ids = test['id']
process_test_text = preprocess(test_text)
X_test = tfidf.fit_transform(process_test_text).toarray()
y_submission_pred = clf.predict(X_test)
sub = pd.Series(y_submission_pred)
data = {'id':ids,

        'target':sub}



final = pd.DataFrame(data)
final.to_csv('submission.csv', index=False)