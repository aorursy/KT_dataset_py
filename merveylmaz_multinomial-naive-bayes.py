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
# Kütüphanelerin Tanımlanması

import numpy as np

import pandas as pd

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix

from sklearn import metrics
# Dataset Okuma / Feature ve Target Oluşturma / Train ve Test Ayırma

heart = pd.read_csv("../input/heart-disease-uci/heart.csv")

X = pd.DataFrame(heart.iloc[:,:-1]) 

y = pd.DataFrame(heart.iloc[:,-1])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Multinomial Naive Bayes için Eğitim ve Tahmin

mnb = MultinomialNB()

y_pred = mnb.fit(X_train, y_train).predict(X_test)
# Multinomial Naive Bayes için Accuracy ve Confusion Matrix Hesaplama

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))