# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

from sklearn.naive_bayes import GaussianNB

import numpy as np
data=pd.DataFrame()

data['outlook']=['sunny','sunny','overcast','rainy','rainy','rainy','overcast','sunny','sunny','rainy','sunny','overcast','overcast','rainy']

data['temp']=['hot','hot','hot','mild','cool','cool','cool','mild','cool','mild','mild','mild','hot','mild',]

data['humidity']=['high','high','high','high','normal','normal','normal','high','normal','normal','normal','high','normal','high']

data['windy']=['FALSE','TRUE','FALSE','FALSE','FALSE','TRUE','TRUE','FALSE','FALSE','FALSE','TRUE','TRUE','FALSE','TRUE']

data['play']=['no','no','yes','yes','yes','no','yes','no','yes','yes','yes','yes','yes','no']

data
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



#data['outlook']=data.iloc[:,0]

data['outlook'] = le.fit_transform(data['outlook'])



#data['temp']=data.iloc[:,1]

data['temp']=le.fit_transform(data['temp'])



#data['humidity']=data.iloc[:,2]

data['humidity']=le.fit_transform(data['humidity'])



#data['windy']=data.iloc[:,3]

data['windy']=le.fit_transform(data['windy'])



data['play']=data.iloc[:,4]

data['play']=le.fit_transform(data['play'])

X = data.iloc[:, :-1].values

y = data.iloc[:, 4].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
classifier=GaussianNB()

classifier.fit(X_train, y_train)
predicted = classifier.predict(X_test)

print(predicted)
from sklearn.metrics import confusion_matrix, accuracy_score

accuracy = accuracy_score(y_test, predicted)

print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
cm = confusion_matrix(y_test, predicted)

cm