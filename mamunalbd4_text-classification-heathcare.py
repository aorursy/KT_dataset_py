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

import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
df = pd.read_csv('/kaggle/input/text-classificationheathcare/TextClassification_Data.csv', encoding='latin')
df.head()
df = df[['SUMMARY', 'categories']]
df.head()
df['categories'].value_counts()
all_mind = {'PRESCRIPTION': 'PRESCRIPTION', 'APPOINTMENTS': 'APPOINTMENTS', 'MISCELLANEOUS': 'MISCELLANEOUS', 'mISCELLANEOUS': 'MISCELLANEOUS', 'JUNK': 'MISCELLANEOUS', 'ASK_A_DOCTOR':'ASK_A_DOCTOR', 'asK_A_DOCTOR': 'ASK_A_DOCTOR', 'LAB':'LAB' }
df['categories'] = [all_mind[x] for x in df['categories']]
df.head()
df['categories'].value_counts()
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
plt.figure(figsize=(20,12))

sns.countplot(x = 'categories', data =df)
df.head()
from sklearn.model_selection import train_test_split
X = df['SUMMARY']

y = df['categories']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
text_cla = Pipeline([('tfid', TfidfVectorizer()), ('clas', LinearSVC())])
text_cla.fit(X_train, y_train)
prediction = text_cla.predict(X_test)
from sklearn import metrics
print(metrics.classification_report(y_test, prediction))
print(metrics.accuracy_score(y_test, prediction))
text_cla.predict(['please call doctor'])
text_cla.predict(['lab report'])