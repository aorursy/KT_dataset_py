P=predictions>0.5

H=P.astype(np.int)

output=np.zeros((418,2))

for i in range(418):

    output[i,0]=ID[i]

    output[i,1]=H[i][0]

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense

a=pd.read_csv("../input/titanic/titanic/train.csv")

b=pd.read_csv("../input/titanic/titanic/test.csv")

#print(a)

a = a.fillna(0)

data=a.values

S = data[:,1]

C = data[:,2]

Age = data[:,5]

G = data[:,4]





Gnum = pd.get_dummies(G)

F = Gnum.values

male = F[:,1]

female = F[:,0]

X = [C,Age,female,male]

Inp = np.transpose(X)



b = b.fillna(0)

testdata=b.values

Ct = testdata[:,1]

Aget = testdata[:,4]

Gt = testdata[:,3]





Gnumt = pd.get_dummies(Gt)

Ft = Gnumt.values

malet = Ft[:,1]

femalet = Ft[:,0]

Xt = [Ct,Aget,femalet,malet]

Inpt = np.transpose(Xt)

ID = testdata[:,0]





# create model

model = Sequential()

model.add(Dense(5, input_dim=4, activation='relu'))

model.add(Dense(7, activation='relu'))

model.add(Dense(1, activation='sigmoid'))



# Compile model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# Fit the model

model.fit(Inp, S, epochs=150, batch_size=20)



#Evaluate

scores = model.evaluate(Inp, S)

#print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

predictions = model.predict(Inpt)













# Any results you write to the current directory are saved as output.
ans=output.astype(np.int)

#new=np.zeros((419,1))

new = pd.DataFrame({'PassengerId':ans[:,0],'Survived':ans[:,1]})

new

new.to_csv('mysubmission.csv',index=False)

new    

    