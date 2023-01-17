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
df = pd.read_csv('/kaggle/input/social-network-ads/Social_Network_Ads.csv')

df.info()
from sklearn.tree import DecisionTreeClassifier

from sklearn import model_selection

from sklearn.preprocessing import LabelEncoder, StandardScaler

#split data into X, y

X = df.iloc[:, 1 :4]

y = df.iloc[:, -1]

label_encoder = LabelEncoder()

X['Gender'] = label_encoder.fit_transform(X['Gender'])

X.head()


#Split data into Training set and Test set

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=10)

#Create Decision Tree classifier

clf = DecisionTreeClassifier()

clf.fit(X_train, y_train)



y_pred = clf.predict(X_test)



from sklearn import metrics

print('Accuracy Score:', metrics.accuracy_score(y_test,y_pred))