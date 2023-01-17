# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import GaussianNB

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/spam-mails-dataset/spam_ham_dataset.csv")



print(df)
x = df.text

y = df.label_num

x_train, x_text, y_train, y_test =  train_test_split(x,y)
vectorize =  CountVectorizer()

counts =  vectorize.fit_transform(x_train.values)

classifierG = GaussianNB()

classifierG.fit(counts.toarray(), y_train.values)

classifierM = MultinomialNB()

classifierM.fit(counts, y_train.values)
classifierG.score( vectorize.transform(x_text).toarray(), y_test)
classifierM.score( vectorize.transform(x_text), y_test)