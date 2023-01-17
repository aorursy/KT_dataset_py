# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

%matplotlib inline
data = pd.read_csv('../input/diabetes.csv',delimiter=',')
data.head()
data.info()
data.describe()
data.shape
dataset = data.values

# X = dataset[:,0:8]

# Y=dataset[:,8]

X = data.drop('Outcome',axis=1).values
type(X)
Y = data['Outcome'].values
type(Y)
Outcome = data['Outcome'].value_counts().values

labels = ['zero','one']

# making pie chart for Outcome through matplotlib

fig1,ax1 = plt.subplots() # fig1 is the window in which

ax1.pie(Outcome,labels=labels,autopct='%.2f%%')

plt.show()
sns.lineplot('Outcome','Age',data=data)
sns.barplot('Outcome','Age',data=data)
sns.pairplot(data,hue='Outcome')
sns.pointplot(data['Outcome'],data['Age'],data=data)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.50,random_state=3)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,Y_train)
score = lm.score(X_test,Y_test)

print('%.2f%%'%(score))
from keras.models import Sequential

from keras.layers import Dense
valid_score = []
from sklearn.model_selection import StratifiedKFold



kf = StratifiedKFold(n_splits=10,shuffle=True,random_state=3)

overall_score=[]



for train, test in kf.split(X,Y):

    model = Sequential()

    model.add(Dense(9,input_dim=8,activation='relu'))

    model.add(Dense(3,activation='relu'))

    model.add(Dense(1,activation="sigmoid"))

    

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X[train],Y[train],verbose=0,epochs=150,batch_size=10)

    

    score = model.evaluate(X[test],Y[test],verbose=0)

    print("Accuracy: %.2f%%" % (score[1]*100))

    overall_score.append(score[1]*100)

    

print("Result: %.2f%% (+/- %.2f%%)" % (np.mean(overall_score),np.std(overall_score)))

    