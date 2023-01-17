# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

from IPython.display import display, Markdown

from itertools import islice
smses = pd.read_csv('../input/SMSSpamCollection', sep = '\t', header=None, names = ['label', 'content'])

smses.head()
Markdown('''There are __{rows}__ data points'''.format(rows=smses.shape[0]))
smses.info()
count_df = smses['label'].value_counts().to_frame()

display(count_df)

sns.barplot(x=count_df.index, y='label', data=count_df)
spam_percentage = '%.2f' % ((count_df.loc['spam'].array[0] / smses.shape[0])*100)
Markdown('''There are __{spam}%__ sms reported as spam '''.format(spam=spam_percentage))
from nltk.corpus import stopwords

english_stopwords = stopwords.words('english')
from nltk.corpus import stopwords

from nltk import word_tokenize

def remove_stopwords(message):

    message = message.lower()

    

    # tokenize

    words = word_tokenize(message);

    

    # remove stop words

    words = list(filter(lambda w: w not in english_stopwords, words))

    

    return ' '.join(words);


# remove stop words

smses['content'] = smses['content'].apply(remove_stopwords)
smses['label'] = smses['label'].map({

    'spam': 1,

    'ham': 0

})
smses.head()
from sklearn.model_selection import train_test_split

smses_train, smses_test = train_test_split(smses,train_size=.7, stratify=smses['label'], random_state=100)



print('training dataset size: ', smses_train.shape)

print('test dataset size: ', smses_test.shape)
Markdown('''

- Spam percentage of dataset is `{}`%

- Spam percentage of Training dataset is `{}`% 

- Spam percentage of Test dataset is `{}`%

'''.format(spam_percentage,

         '%.2f' % (smses_train['label'].mean() * 100),

         '%.2f' % (smses_test['label'].mean() * 100)))
from sklearn.feature_extraction.text import CountVectorizer



# with removal of stop words

vec = CountVectorizer()
vec.fit(smses_train['content'])



# first 5 item in vocabulary

list(islice(vec.vocabulary_.items(), 5))
X = vec.transform(smses_train['content'])

y = smses_train['label']



X_test = vec.transform(smses_test['content'])

y_test = smses_test['label']
from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()
mnb.fit(X, y)
from sklearn import metrics
y_test_pred = mnb.predict(X_test)

Markdown('''The test data accuracy is __{accurancy}__'''.format(accurancy = '%.2f' % metrics.accuracy_score(y_test, y_test_pred)))
cm = metrics.confusion_matrix(y_test, y_test_pred)
# Thus in binary classification, the count of true negatives is

#    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is

#    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.



# correctly predicted positive values

true_positive = cm[1, 1] 



# correctly predicted negative values

true_negative = cm[0, 0]



# incorrectly predicted positive values

false_positive = cm[0, 1]



# incorrecly predicted negative values

false_negative =  cm[1, 0]
sensitivity = true_positive / (true_positive + false_negative)



Markdown('''The sensitivity of model is __{sensitivity}__'''.format(sensitivity = '%.2f' % sensitivity))
specificity = true_negative / (true_negative + false_positive)



Markdown('''The specificity of model is __{specificity}__'''.format(specificity = '%.2f' % specificity))
precision = true_positive / (true_positive + false_positive)



Markdown('''The precision of model is __{precision}__'''.format(precision = '%.2f' % precision))
f1_score =  2 * ((precision * sensitivity)/(precision + sensitivity))

Markdown('''The f Score of model is __{f1_score}__'''.format(f1_score = '%.2f' % precision))
y_test_proba = mnb.predict_proba(X_test)[:, -1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_test_proba)
import matplotlib.pyplot as plt

plt.xlabel('True Postive Rate')

plt.ylabel('False Postive Rate')

plt.title('ROC Curve')

plt.plot(fpr, tpr)
auc = metrics.auc(fpr, tpr)

Markdown('''Area under the curve is __{auc}__'''.format(auc = '%.2f' % auc))