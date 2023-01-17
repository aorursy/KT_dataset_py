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
df = pd.read_csv('../input/mobile-price-classification/train.csv')

df
target_col = 'price_range'
df[target_col].unique()
df[target_col].value_counts()
y = df[target_col]

X = df.drop(columns=[target_col])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)
para = list(range(50, 1001, 50))

print(para)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, accuracy_score, f1_score

results = {}

for n in para:

    print('para=', n)

    model = RandomForestClassifier(n_estimators=n)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    accu = accuracy_score(y_true=y_test, y_pred=preds)

    f1 = f1_score(y_true=y_test, y_pred=preds,  average = 'micro')

    print(classification_report(y_true=y_test, y_pred=preds))

    print('--------------------------')

    results[n] = f1
results
import matplotlib.pylab as plt

# sorted by key, return a list of tuples

lists = sorted(results.items()) 

p, a = zip(*lists) # unpack a list of pairs into two tuples

plt.plot(p, a)

plt.show()
final_model = RandomForestClassifier(n_estimators=800)

final_model.fit(X, y)
test_df = pd.read_csv('../input/mobile-price-classification/test.csv', index_col='id')

test_df 
X.shape
test_df.shape
test_df.columns
set(test_df.columns) - set(df.columns)
test_preds = final_model.predict(test_df)