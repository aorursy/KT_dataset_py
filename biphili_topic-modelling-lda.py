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
npr=pd.read_csv('../input/topic-modeling/npr.csv')

npr.head()
npr.shape
npr['Article'][0]
len(npr)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_df=0.9,min_df=2,stop_words='english')
dtm=cv.fit_transform(npr['Article'])
dtm
from sklearn.decomposition import LatentDirichletAllocation
LDA=LatentDirichletAllocation(n_components=7,random_state=42)
LDA.fit(dtm)
len(cv.get_feature_names())
type(cv.get_feature_names())
cv.get_feature_names()[41000]
import random

random_word_id=random.randint(0,54777)

cv.get_feature_names()[random_word_id]
len(LDA.components_)
LDA.components_.shape
LDA.components_
single_topic=LDA.components_[0]
single_topic.argsort()
import numpy as np

arr=np.array([10,200,1])
arr
arr.argsort()
# Argsort gives index in ascending order 

single_topic.argsort()[-10:]
top_ten_words=single_topic.argsort()[-10:]
for index in top_ten_words:

    print(cv.get_feature_names()[index])
for i,topic in enumerate(LDA.components_):

    print(f"THE TOP 15 WORDS FOR THE TOPIC #{i}")

    print([cv.get_feature_names()[index] for index in topic.argsort()[-15:]])

    print('\n')

    print('\n')
topic_results=LDA.transform(dtm)
topic_results.shape
topic_results[0]
topic_results[0].round(2)
npr['Article'][0]
topic_results[0].argmax()
npr['Topic']=topic_results.argmax(axis=1)
npr
mytopic_dict={0:'Health',1:'Election',2:'Family',3:'Politics',4:'Election',5:'Music',6:'Education'}

npr['Topic Label']=npr['Topic'].map(mytopic_dict)
npr