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
#Importing Libraries

import pandas as pd



#Importing the Dataset

names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

dataset = pd.read_csv("../input/iris.dataset.csv", names = names)
#Data Preprocessing

X = dataset.iloc[:, 0:4].values

y = dataset.iloc[:,4].values


from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#Feature Scaling

from sklearn.preprocessing import StandardScaler



ss= StandardScaler()

X_train = ss.fit_transform(X_train)

X_test = ss.transform(X_test)
#Performing LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



lda = LDA(n_components=1)

X_train = lda.fit_transform(X_train, y_train)

X_test = lda.transform(X_test)
#Training and Making Predictions

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=2, random_state=0)



model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Evaluating the Performance

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_pred)

print(cm)



print('Accuracy' + str(accuracy_score(y_test, y_pred)))