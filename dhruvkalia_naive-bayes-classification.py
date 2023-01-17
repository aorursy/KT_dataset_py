# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import classification_report, plot_confusion_matrix

import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import OneHotEncoder



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/gender-classification/Transformed Data Set - Sheet1.csv")

df.describe()
X = df.iloc[:, 0:4]

y = df.iloc[:, 4]
clf = Pipeline(steps=[

    ("onehot", OneHotEncoder(handle_unknown='ignore', sparse=False)),

    ("scaler", StandardScaler()),

    ("model", GaussianNB()),

])
clf.fit(X, y)

print(classification_report(y, clf.predict(X)))

print(plot_confusion_matrix(clf, X, y, cmap=plt.cm.Blues,

                            display_labels=['Male', 'Female']))
