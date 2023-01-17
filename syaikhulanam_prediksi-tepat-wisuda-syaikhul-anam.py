# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from sklearn import preprocessing

#from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix

from sklearn import metrics

from sklearn.metrics import plot_confusion_matrix

from sklearn import svm

from sklearn.multiclass import OneVsOneClassifier
dataset = pd.read_csv('/kaggle/input/on-time-graduation-classification/data_lulus_tepat_waktu.csv')
X = dataset.drop(['tepat'],axis=1)

y = dataset['tepat']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.25, random_state=0)

klasifier = svm.SVC(kernel='linear', cache_size=1000, class_weight='balanced')
klasifierfit = OneVsOneClassifier(klasifier).fit(X_train, y_train)
hasil_prediksi = klasifierfit.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print(confusion_matrix(y_test, hasil_prediksi))

print(classification_report(y_test, hasil_prediksi))

print(accuracy_score(y_test, hasil_prediksi))