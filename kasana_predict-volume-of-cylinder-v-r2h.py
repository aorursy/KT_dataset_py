



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

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



dfTrain = pd.read_csv("../input/CylinderVolumeData.csv")

dfTrain.head()
dfTrain.plot(x='radius',y='volume',kind='scatter')

plt.show()
dfTrain.plot(x='height',y='volume',kind='scatter')

plt.show()
#%matplotlib notebook

plt3D = plt.figure().gca(projection='3d')

plt3D.scatter(dfTrain['radius'], dfTrain['height'], dfTrain['volume'])

plt3D.set_xlabel('radius')

plt3D.set_ylabel('height')

plt3D.set_zlabel('volume')

plt.show()
#MapFeature - This function is used to provide 2 level degree to the independent variables

def mapFeature(X,degree):

    

    sz=X.shape[1]

    if (sz==2):

        sz=(degree+1)*(degree+2)/2

        sz=int(sz)

    else:

         sz=degree+1

    out=np.ones((X.shape[0],sz))     #Adding Bias W0



    sz=X.shape[1]

    if (sz==2):

        X1=X[:, 0:1]

        X2=X[:, 1:2]

        col=1

        for i in range(1,degree+1):        

            for j in range(0,i+1):

                out[:,col:col+1]= np.multiply(np.power(X1,i-j),np.power(X2,j))    

                col+=1

        return out

    else:

        for i in range(1,degree+1):        

            out[:,i:i+1]= np.power(X,i)

    

    return out
import sklearn.linear_model  as LR



df_Features=dfTrain.iloc[:,0:2]

df_Label=dfTrain.iloc[:,2:3]



X = df_Features.values

Y = df_Label.values.ravel()
#Standard Scaling - Data Preprocessing



#Data Preprocessing

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(X)

XS = scaler.transform(X)

inputX = mapFeature(XS,3) 
RegObj = LR.SGDRegressor(learning_rate='adaptive',eta0=0.1,alpha=0,max_iter=1000).fit(inputX,Y)

pred = RegObj.predict(mapFeature(scaler.transform([[189,177]]),3) )

print(pred)
#You can obtain the coefficient of determination (𝑅²) with .score() called on model:

#When you’re applying .score(), the arguments are also the predictor x and regressor y, and the return value is 𝑅².

print('coefficient of determination:', RegObj.score(inputX,Y))

#The attributes of model are .intercept_, which represents the coefficient, 𝑏₀ and .coef_, which represents 𝑏₁:

print('intercept:', RegObj.intercept_)

print('slope:', RegObj.coef_)
x_min, x_max = X[:, 0].min() , X[:, 0].max() 

y_min, y_max = X[:, 1].min() , X[:, 1].max() 

u = np.linspace(x_min, x_max,10) 

v = np.linspace(y_min, y_max, 10) 

z = np.zeros(( len(u), len(v) )) 

U,V=np.meshgrid(u,v)

for i in range(len(u)): 

    for j in range(len(v)): 

        uv= np.column_stack((np.array([[u[i]]]),np.array([[v[j]]])))

        pred = RegObj.predict(mapFeature(scaler.transform(uv),3) )

     

        z[i,j] =pred[0]

z = np.transpose(z) 
#%matplotlib notebook

plt3D = plt.figure().gca(projection='3d')

plt3D.scatter(dfTrain['radius'], dfTrain['height'], dfTrain['volume'])

plt3D.scatter(U,V,z)

plt3D.set_xlabel('radius')

plt3D.set_ylabel('height')

plt3D.set_zlabel('volume')

plt.show()
#Contour Graph 

plt3D = plt.figure().gca(projection='3d')

plt3D.scatter(dfTrain['radius'], dfTrain['height'], dfTrain['volume'],alpha=0.01,color='r')

plt3D.contourf(U,V,z,color='b',alpha=0.6)

plt3D.set_xlabel('radius')

plt3D.set_ylabel('height')

plt3D.set_zlabel('volume')

plt.show()
import pickle

with open('SGDLearning.pkl', 'wb') as f:

  pickle.dump(RegObj,f,pickle.HIGHEST_PROTOCOL)
with open('SGDLearning.pkl', 'rb') as f:

  RegObj = pickle.load(f)
PredictionFromFile = RegObj.predict(mapFeature(scaler.transform([[189,177]]),3) )

print(PredictionFromFile)