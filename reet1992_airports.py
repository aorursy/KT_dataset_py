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

import seaborn as sns

import matplotlib.pyplot as plt

import sklearn
df = pd.read_csv('../input/usa-airport-dataset/Airports2.csv')
type(df)
df
data = df
label = data['Distance']/data['Flights']
label
type(label)
label = pd.DataFrame(label)
data['profit'] = label
data
len(data)
data2 = data.dropna()
len(data)
len(data2)
lable = data2['profit']
data2['label'] = lable
data2
data2.describe()
data3 = data2.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
data3 = data2[data2.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
data3.describe()
print(data3['label'].max())

print(data3['label'].min())
plt.hist(data3['label'])
data3['label'].loc[(data3['label'] < 50)] = 0

data3['label'].loc[(data3['label'] >= 50) & (data3['label'] < 100)] = 1

data3['label'].loc[(data3['label'] >= 100) & (data3['label'] < 200)] = 2

data3['label'].loc[(data3['label'] >= 200) & (data3['label'] < 500)] = 3

data3['label'].loc[(data3['label'] >= 500) & (data3['label'] < 1000)] = 4

data3['label'].loc[(data3['label'] >= 1000) & (data3['label'] < 2000)] = 5

data3['label'].loc[(data3['label'] > 2000)] = 6
data3
data3.info()
type(data3['Origin_population'])
label2 = data3['label'].astype('category')
# Classification Problem
data3.head()
sns.heatmap(data3.corr())
x = data3[['Distance','Origin_population','Destination_population']]

x = x.values

y = data3[['label']]

y = y.values
x.shape
y.shape
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=1)

clf = DecisionTreeClassifier(max_depth=None, max_leaf_nodes=10)

clf
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_pred
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred,y_test)

cm
plt.matshow(cm)

plt.title('Confusion matrix')

plt.colorbar()

plt.ylabel('True label')

plt.xlabel('Predicted label')

plt.show()