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

import re

import plotly.express as px

import seaborn as sns

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

stopwords = set(STOPWORDS)

import collections

import nltk

from nltk.tokenize import sent_tokenize,word_tokenize

import matplotlib.pyplot as plt 

import seaborn as sns 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics



nltk.download('punkt')
train = pd.read_csv('/kaggle/input/training-meta-info-nlp-tweets/train_v3.csv')

test = pd.read_csv('/kaggle/input/tweetstest-v3/test_v3.csv')

train['text'] = train['text'].astype(str)

test['text'] = test['text'].astype(str)
train
feature_cols = ['Number_of_words',	'Number_of_Sentences',	'Number_of_Unique_Words',	'Number_of_Stop_Words',	'Number_of_Hashtage',	'Number_of_Mentions',	'Average_Word_Length']
X = train[feature_cols] # Features

y = train.target
# split X and y into training and testing sets

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)
# instantiate the model (using the default parameters)

logreg = LogisticRegression()



# fit the model with data

logreg.fit(X_train,y_train)



#

y_pred=logreg.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

cnf_matrix

class_names=[0,1] # name  of classes

fig, ax = plt.subplots()

tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names)

plt.yticks(tick_marks, class_names)

# create heatmap

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

ax.xaxis.set_label_position("top")

plt.tight_layout()

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))
y_pred_proba = logreg.predict_proba(X_test)[::,1]

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)

auc = metrics.roc_auc_score(y_test, y_pred_proba)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))

plt.legend(loc=4)

plt.show()
def threshold_prob(predictions,threshold):

  arr = []

  for p in predictions:

    if p>threshold:

      arr.append(1)

    else:

      arr.append(0)

  return arr





predicted_class = threshold_prob(logreg.predict_proba(X_test)[::,1],0.3)
cnf_matrix = metrics.confusion_matrix(y_test, predicted_class)

cnf_matrix

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Precision:",metrics.precision_score(y_test, y_pred))

print("Recall:",metrics.recall_score(y_test, y_pred))