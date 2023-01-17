import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split as tts

%matplotlib inline
X=[]
Y=[]



def generator(x):
    noise = np.random.rand()*7
    if x<=8 and x>=-8:
        return 64-x*x + noise 
    elif x<-8:
        return 8*np.sqrt(-8-x) + noise 
    else:
        return 8*np.sqrt(x-8) + noise 

X=[]
Y=[]
Z=[]
y=0
x=0
sample = np.linspace(-20,20, 1000)
for x in sample:
    noise = np.random.rand()*20
    y = generator(x)
    z = 10
    X.append(x)
    Y.append(y)
    Z.append(z + y + noise)
X= np.array(X)
Y= np.array(Y)
generator(0)
%matplotlib inline
sns.scatterplot(X,Y)
plt.show()
from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(plt.figure())
ax.scatter(X,Y,Z)
ax
print(Y.shape)
XData2= np.append(X.reshape(-1,1),Y.reshape(-1,1),axis = 1)
YData2 = np.array(Z).T

XData2.shape,YData2.shape
XData2[:10], YData2[:10]
xtrain,xtest,ytrain,ytest = tts(XData2,YData2, random_state = 2, shuffle=True, train_size = 0.7)
print(xtrain.shape,ytrain.shape)
# train = np.vstack((xtrain,ytrain.reshape(-1,1)))
train= np.append(xtrain,ytrain.reshape(-1,1), axis =1 )
test = xtest
testLabels = ytest


df = pd.DataFrame(train)
df.columns = ["X","Y","Z"]

df1 = pd.DataFrame(test)
df1.columns = ["X", "Y"]
df1.head()

df2 = pd.DataFrame(testLabels)
df2.columns = ["Y"]
df2.head()

df3 = pd.DataFrame(np.zeros((len(testLabels),1)))
df3.columns = ["Y"]
df3.head()

print(df.shape, df1.shape, df2.shape, df3.shape)

df.to_csv("trainData3D.csv",sep=",",header = True, index = False) # because data cleaning is not the objective
df1.to_csv("testData3D.csv", sep=",", header = True, index = False)
df2.to_csv("actualYTest3D.csv", sep=",", header = True, index = False)
df3.to_csv("samplesubmission3D.csv", sep=",", header = True, index = False)

%matplotlib inline

from mpl_toolkits.mplot3d import Axes3D
ax = Axes3D(plt.figure())
ax.scatter(X,Y,Z, color = "orange", marker = "^")
ax
xtrain.shape,xtest.shape
xtrain,xtest,ytrain,ytest = tts(X,Y, random_state = 2, shuffle=True, train_size = 0.7)

train = np.vstack((xtrain,ytrain)).T
test = xtest.T
testLabels = np.array(ytest).T

df = pd.DataFrame(train)
df.columns = ["X","Y"]

df1 = pd.DataFrame(test)
df1.columns = ["X"]
df1.head()

df2 = pd.DataFrame(testLabels)
df2.columns = ["Y"]
df2.head()

df3 = pd.DataFrame(np.zeros((len(testLabels),1)))
df3.columns = ["Y"]
df3.head()


df.to_csv("train.csv",sep=",",header = True, index = False) # because data cleaning is not the objective
df1.to_csv("test.csv", sep=",", header = True, index = False)
df2.to_csv("actualYTest.csv", sep=",", header = True, index = False)
df3.to_csv("samplesubmission.csv", sep=",", header = True, index = False)

df3.shape,df2.shape,df1.shape

%matplotlib inline
sns.scatterplot(xtrain,ytrain)
sns.scatterplot(xtest,ytest)
plt.legend()
