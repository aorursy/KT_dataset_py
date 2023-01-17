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
import pandas as pd  

import numpy as np  



data = pd.read_csv('../input/fifa19/data.csv')
data.info()
data.drop('Unnamed: 0' , axis= 1 , inplace = True)
#data = data.replace(to_replace = np.nan , value =0)
x = data['Name']

y = data['Position'].astype(str)
from sklearn.model_selection import train_test_split  

x_train , x_test , y_train , y_test = train_test_split(x , y , random_state = 0)
from sklearn.feature_extraction.text import CountVectorizer



# Fit the CountVectorizer to the training data

vect = CountVectorizer().fit(x_train)
# transform the documents in the training data to a document-term matrix

X_train_vectorized = vect.transform(x_train)

X_train_vectorized
X_train_vectorized
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train_vectorized , y_train)

knn.score(X_train_vectorized , y_train)
print(knn.predict(vect.transform(['F. Rodríguez'])))