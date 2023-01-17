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
missing_data = pd.read_excel('/kaggle/input/missing_data.xlsx',header=None)

interpolate_data = pd.read_excel('/kaggle/input/missing_data.xlsx',header=None)

missing_data
from scipy.interpolate import lagrange



def ployinterp_column(s,n,k=5):

  y = s[list(range(n-k,n)) + list(range(n+1,n+1+k))] 

  y = y[y.notnull()] 

  return lagrange(y.index,list(y))(n) 



for i in missing_data.columns:

  for j in range(len(missing_data)):

    if (missing_data[i].isnull())[j]: 

      missing_data[i][j] = ployinterp_column(missing_data[i],j)



missing_data.head(10)
print(missing_data[3:4])

print(missing_data[4:5])

print(missing_data[5:6])

print(missing_data[7:8])

print(missing_data[10:11])

print(missing_data[12:13])
data = interpolate_data.interpolate()

print(data[3:4])

print(data[4:5])

print(data[5:6])

print(data[7:8])

print(data[10:11])

print(data[12:13])
model_data = pd.read_excel('/kaggle/input/model.xlsx')

model_data.head()
from random import shuffle

model_data = model_data.as_matrix()

shuffle(model_data)



p = 0.8 

train = model_data[:int(len(model_data)*p),:]

test = model_data[int(len(model_data)*p):,:]
train
test
from sklearn.tree import DecisionTreeClassifier 

from sklearn.metrics import confusion_matrix 



tree = DecisionTreeClassifier() 

tree.fit(train[:,:3],train[:,3]) 





cm = confusion_matrix(train[:,3], tree.predict(train[:,:3])) 



import matplotlib.pyplot as plt 

plt.matshow(cm, cmap=plt.cm.Greens) 

plt.colorbar() 



for x in range(len(cm)): 

  for y in range(len(cm)):

    plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')



plt.ylabel('True label') 

plt.xlabel('Predicted label') 

plt.show() 
from sklearn.metrics import roc_curve 



fpr, tpr, thresholds = roc_curve(test[:,3], tree.predict_proba(test[:,:3])[:,1], pos_label=1)

plt.plot(fpr, tpr, linewidth=2, label = 'ROC of CART', color = 'green') 

plt.xlabel('False Positive Rate') 

plt.ylabel('True Positive Rate') 

plt.ylim(0,1.05) 

plt.xlim(0,1.05) 

plt.legend(loc=4)

plt.show() 
from keras.models import Sequential 

from keras.layers.core import Dense,Activation 



net = Sequential()

net.add(Dense(10,input_dim=3)) # units=10,input_dim =3

net.add(Activation('relu'))

net.add(Dense(1,input_dim=10)) # units=1,input_dim =10

net.add(Activation('sigmoid'))

net.compile(loss='binary_crossentropy',optimizer = 'adam')



net.fit(train[:,:3], train[:,3], nb_epoch=1000, batch_size=1) 



from sklearn.metrics import confusion_matrix 



predict_result = net.predict_classes(train[:,:3]).reshape(len(train)) 





cm = confusion_matrix(train[:,3], predict_result) 



import matplotlib.pyplot as plt 

plt.matshow(cm, cmap=plt.cm.Greens) 

plt.colorbar() 



for x in range(len(cm)): 

  for y in range(len(cm)):

    plt.annotate(cm[x,y], xy=(x, y), horizontalalignment='center', verticalalignment='center')



plt.ylabel('True label') 

plt.xlabel('Predicted label') 

plt.show() #显示作图结果



from sklearn.metrics import roc_curve



predict_result = net.predict(test[:,:3]).reshape(len(test))

fpr, tpr, thresholds = roc_curve(test[:,3], predict_result, pos_label=1)

plt.plot(fpr, tpr, linewidth=2, label = 'ROC of LM') 

plt.xlabel('False Positive Rate') 

plt.ylabel('True Positive Rate') 

plt.ylim(0,1.05)

plt.xlim(0,1.05) 

plt.legend(loc=4)

plt.show() 
