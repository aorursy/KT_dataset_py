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
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
simple_train = pd.read_csv("../input/train.csv")

simple_train.head()
X = simple_train.comment_text
y = simple_train.toxic
print (X.shape)
print (y.shape)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=5,test_size=.5)
vect=CountVectorizer(stop_words='english')
X_train_dtm=vect.fit_transform(X_train)
X_train_dtm
X_test_dtm = vect.transform(X_test)
X_train_tokens= vect.get_feature_names()
X_train_counts = np.sum(X_train_dtm.toarray(),axis=0)
X_train_counts
new_df = pd.DataFrame({"token":X_train_tokens,"count":X_train_counts})
new_df.sort_values(by='count',ascending=False).head()
logreg= LogisticRegression()
logreg.fit(X_train_dtm,y_train)
y_pred_class = logreg.predict(X_test_dtm)
from sklearn import metrics
metrics.accuracy_score(y_test,y_pred_class)
metrics.confusion_matrix(y_test,y_pred_class)
X_test[y_test<y_pred_class]
