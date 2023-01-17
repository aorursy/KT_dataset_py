# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder



from sklearn.model_selection import train_test_split



from sklearn.naive_bayes import MultinomialNB



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/Articles.csv", encoding = "ISO-8859-1")
data.head()
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(data['Article'])



encoder = LabelEncoder()

y = encoder.fit_transform(data['NewsType'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
gnb = MultinomialNB()

%time gnb.fit(X_train, y_train)

print('Correct % for Naive Bayes : ' + str(gnb.score(X_test, y_test)*100)+ ' %')



print('<========================================================>\n')