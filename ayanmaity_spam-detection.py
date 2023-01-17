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
import textblob
df = pd.read_csv("../input/spam.csv",encoding='latin-1')
df.columns
df.head()
del df['Unnamed: 2']

del df['Unnamed: 3']

del df['Unnamed: 4']
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

c_vec = CountVectorizer(lowercase=1,min_df=.00001,stop_words='english')

c_vec.fit(df['v2'].values)

X_train = c_vec.transform(df['v2'][0:5000].values)
Y_train = df['v1'][0:5000].values

clf_nb = MultinomialNB()

clf_nb.fit(X_train,Y_train)

clf_nb.score(X_train,Y_train)
X_test = c_vec.transform(df['v2'][5000:].values)

Y_test = df['v1'][5000:].values

clf_nb.score(X_test,Y_test)