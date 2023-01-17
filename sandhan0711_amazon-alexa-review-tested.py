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
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data=pd.read_csv('../input/amazon_alexa.tsv',delimiter='\t',quoting=3)
data
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
ps=PorterStemmer()
corpus=[]
import re
for i in range(0,3150):
    review=re.sub('[^a-zA-Z]',' ',data.values[i][3])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)
x=cv.fit_transform(corpus).toarray()
y=data.iloc[:,0].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=20, criterion='entropy', random_state=0)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
y_pred
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_pred,y_test)
cm
y
new_entry="Alexa is simply amazing, it is a smashing product and I absolutely adore it!!!"
new_entry
new_review = re.sub("[^a-zA-Z]", " ", new_entry)    
new_review = new_review.lower().split()
new_review = [ps.stem(word) for word in new_review if word not in set(stopwords.words("english"))]    
new_review = " ".join(new_review)    
new_review = [new_review]    
new_review = cv.transform(new_review).toarray()
new_pred=classifier.predict(new_review)[0]
new_pred
