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
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

df = pd.read_csv('/kaggle/input/pimaindiansdiabetescsv/pima-indians-diabetes.csv', names = ['pregnancies', 'clucose', 'blood_pressure', 'skin_thinckness', 'insulin', 'bmi', 'diabetes_pedigree', 'extra', 'result'])

X = df.drop(['result'], 1)

y = df['result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)

gaussinan_clf = GaussianNB()

gaussinan_clf.fit(X_train, y_train)

print(gaussinan_clf.score(X_test, y_test))
