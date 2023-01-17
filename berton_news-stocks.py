from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd

import numpy as np

from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score

from datetime import date# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd

import numpy as np

from sklearn.svm import SVC

from sklearn.metrics import roc_auc_score

from datetime import date
data = pd.read_csv('../input/Combined_News_DJIA.csv')

data.head()

data.info()
data["combined_news"] = data.filter(regex=("Top.*")).apply(lambda x: ''.join(str(x.values)), axis=1)
train = data[data['Date'] < '2015-01-01']

test = data[data['Date'] > '2014-12-31']
feature_extraction = TfidfVectorizer()

X_train = feature_extraction.fit_transform(train["combined_news"].values)
X_test = feature_extraction.fit_transform(test["combined_news"].values)
y_train = train["Label"].values

y_test = test["Label"].values
y_train = train["Label"].values

y_test = test["Label"].values
clf = SVC(probability=True, kernel='rbf')
clf.fit(X_train, y_train)
predictions = clf.predict_proba(X_test)
print('ROC-AUC yields ' + str(roc_auc_score(y_test, predictions[:,1])))