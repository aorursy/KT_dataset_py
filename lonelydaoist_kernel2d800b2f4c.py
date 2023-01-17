from shutil import copyfile

copyfile(src="../input/regressionmpdel/regressionModel.py",dst="../working/regressionModel.py")
import regressionModel as rm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
data = pd.read_csv("../input/housesalesprediction/kc_house_data.csv")
data.head()
data.dtypes
data = data.drop(["id","date"],axis=1)

x = data.values
y = np.expand_dims(x[:,0],axis=1)
x = np.delete(x,0,1)
x = rm.preProcess(x)
a = int(x.shape[0] * 0.6)
b = int(x.shape[0] * 0.2)
xt = x[:a]
yt = y[:a]
xc = x[a:a+b]
yc = y[a:a+b]
xte = x[a+b:]
yte = y[a+b:]
theta0 = np.random.random((x.shape[1],1))
l = 0
alphas = [0.001,0.01,0.1]
nIterations = range(200)
for i,alpha in enumerate(alphas,1):
    c = []
    for nIter in tqdm(nIterations):
        t = rm.gradientDescent(xc,yc,theta0,alpha,l,nIter)
        c.append(rm.cost(xc,yc,t,0))
    plt.plot(nIterations,c)
plt.legend(alphas)
plt.show()
c1,c2 = [],[]
nExp = range(1,a)
for i in tqdm(nExp):
  t = rm.gradientDescent(xt[:i],yt[:i],theta0,0.1,l,25)
  c1.append(rm.cost(xc,yc,t,0))
  c2.append(rm.cost(xt,yt,t,0))
plt.plot(nExp,c1)
plt.plot(nExp,c2)
plt.legend(["cross validation set","training set"])
plt.show()
theta = rm.gradientDescent(xt,yt,theta0,0.1,l,25)
print("error: ","{:.2f}".format(np.mean(100 * np.absolute(rm.predict(xc,theta) - yc)/yc)),"%")