# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/train.csv')
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
stopset=set(stopwords.words('english'))
vector=TfidfVectorizer(use_idf=True , lowercase=True , strip_accents='ascii',stop_words=stopset)
vector.fit(data['question_text'])
X=vector.transform(data['question_text'])
X.shape
Y=data['target']
#from sklearn import naive_bayes
#ml=naive_bayes.MultinomialNB()
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='sag',multi_class='ovr')
lr.fit(X,Y)
#ml.fit(X,Y)
test=pd.read_csv('../input/test.csv')
test.shape
voc=vector.transform(test['question_text'])
predict=lr.predict(voc)
print(np.count_nonzero(predict == 1))


df1=pd.DataFrame(test['qid'])
pre={'prediction':predict}
df2=pd.DataFrame(pre)
df1.reset_index(drop=True, inplace=True)
df2.reset_index(drop=True, inplace=True)

result = pd.concat( [df1, df2], axis=1)
result.to_csv('submission.csv',index=False)








