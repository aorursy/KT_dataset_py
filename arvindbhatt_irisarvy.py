import  numpy as np

import pandas as pd

import matplotlib.pyplot as plt

df=pd.read_csv('../input/Iris.csv')

#print(type(df))

df.drop('Id',axis=1,inplace=True)

df['Species']=df.Species.map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})



X=df.loc[:,'SepalLengthCm':'PetalWidthCm']

Y=df['Species']

#plt.plot(X,Y,'ro')

plt.show()





from sklearn import linear_model

import matplotlib.pyplot as plt

reg=linear_model.LogisticRegression()

X=np.array(X)

Y=np.array(Y)



reg.fit(X[0:130,:],Y[0:130])

pred=reg.predict(X[130:,:])

print(reg.score(X[130:,:],Y[130:]))



plt.plot(X[:,0],Y,'ro')

plt.plot(X[130:,0],pred,linewidth=1)

plt.show()



type(Y)
