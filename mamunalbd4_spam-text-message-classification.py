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
import numpy as np

import pandas as pd
df = pd.read_csv('/kaggle/input/spam-text-message-classification/SPAM text message 20170820 - Data.csv')
df.head()
df.isnull().sum()
blanks = []

for i, cat, mes in df.itertuples():

    if type(mes)==str:

        if mes.isspace():

            blanks.append(i)
blanks
from sklearn.model_selection import train_test_split
X = df['Message']

y = df['Category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
text_cla = Pipeline([('tfid', TfidfVectorizer()), ('clf', LinearSVC())])
text_cla.fit(X_train,y_train)
predictions = text_cla.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test, predictions))
print(metrics.confusion_matrix(y_test, predictions))
print(metrics.accuracy_score(y_test, predictions))