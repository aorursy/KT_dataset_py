# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from sklearn.utils import shuffle



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn import metrics

train_df = pd.read_csv('../input/train.csv')

train_df.shape

train_df.head()

train_df['author_num'] = train_df.author.map({'EAP':0, 'HPL':1, 'MWS':2})







train_df=shuffle(train_df)

train_df.head()



testing=pd.read_csv("../input/test.csv")

testing.head()



X = train_df['text']

y = train_df['author_num']

print(X.shape)

print(y.shape)



from nltk.corpus import stopwords

stopset = set(stopwords.words('english'))



vect = CountVectorizer(lowercase=False, token_pattern=r'\w+|\,')

X_cv=vect.fit_transform(X)
from sklearn.naive_bayes import BernoulliNB, MultinomialNB

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import roc_auc_score



from sklearn.ensemble import VotingClassifier



models=[('MultiNB', MultinomialNB(alpha=0.01))]

models.append(('Logit', LogisticRegression(C=30)))

clf1 = VotingClassifier(models, voting='soft', weights=[3,3])

clf1.fit(X_cv,y)

y_pred_test=clf1.predict_proba(X_cv)

all_test=testing["text"]

all_test_v=vect.transform(all_test)

final_result=clf1.predict_proba(all_test_v)
result=pd.DataFrame(final_result)

result.columns = ['EAP', 'HPL', 'MWS']

result.insert(0, 'id', testing["id"])

result.to_csv("resultfinal.csv", index=False)
result.head()