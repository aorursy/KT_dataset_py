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
raw_data=pd.read_csv('/kaggle/input/fraud-email-dataset/fraud_email_.csv')
raw_data.head(2)
raw_data=raw_data[0:4000]
raw_data.info()
raw_data.dropna(inplace=True)
raw_data.info()
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer=TfidfVectorizer()
text_array=vectorizer.fit_transform(raw_data['Text']).toarray()
text_array[0:2]
data=pd.DataFrame(text_array)
data.head()
data['Class']=raw_data['Class']
del raw_data
print(data.head(5))
print(data.info())
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
Y=data.pop('Class')
X=data
x_train,x_test,y_train,y_test=train_test_split(X,Y,shuffle=True)
del X
del Y
model=MLPClassifier(verbose=True,hidden_layer_sizes=(120,120,120,),max_iter=2000)
model.fit(x_train,y_train)


plt.plot(model.loss_curve_)
plt.show()
y_pre_test=model.predict(x_test)
y_pre_train=model.predict(x_train)
print("TEST DATA\n")
print("Accuracy:\n",accuracy_score(y_pre_test,y_test))
print("Report:\n",classification_report(y_pre_test,y_test))
print("Confusion:\n",confusion_matrix(y_pre_test,y_test))
print("\n\nTRAINING DATA\n")
print("Accuracy:\n",accuracy_score(y_pre_train,y_train))
print("Report:\n",classification_report(y_pre_train,y_train))
print("Confusion:\n",confusion_matrix(y_pre_train,y_train))