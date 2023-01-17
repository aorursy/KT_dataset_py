import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))

def stoarray(data = [], sep = ','):
    return data.map(lambda x: np.array(x.split(sep), dtype=float))

# load the data
colna = ['id', 'seq']
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")
test.columns = colna
train.columns = colna
train['seq'], test['seq'] = stoarray(train['seq']), stoarray(test['seq']) 



plt.figure(figsize=(15,10))
plt.suptitle("Train subset", fontsize="x-large")
for k in range(25):
    y = train['seq'][k+1]
    x = np.linspace(1, len(y),len(y))
    plt.subplot(5,5,1+k)
    # plot the points
    plt.scatter(x, y, s=20, c='green', alpha=0.4)
    # shows the expected output
    for j in range(5):
        plt.scatter(x[-1], y[-1], s=10+5**j, c='darkblue', alpha=0.52-0.12*j)
    plt.plot(x, y, c='gray', alpha=0.3)
    plt.axis('off')
plt.savefig('e01.png')

from   sklearn import linear_model

def addpolynomialterms(subX, x):
    subX['x2'] = x**2
    subX['x3'] = x**3
    subX['x4'] = x**.5
    subX['x5'] = np.sin(x)
    subX['x6'] = np.cos(x)
    subX['x7'] = np.exp(x)
    subX['x8'] = np.log(x)

y         = train['seq'][2]
x         = np.linspace(1, len(y),len(y))
regresor  = linear_model.LinearRegression()
subX      = pd.DataFrame({'x':x})
addpolynomialterms(subX, x)
regresor.fit(subX.as_matrix(), y)
predict   = regresor.predict(subX.as_matrix())


plt.figure(figsize=(10,6))
plt.scatter(x, y, s=20, c='green', alpha=0.4)
for j in range(5):
    plt.scatter(x[-1], y[-1], s=50+5**j, c='darkblue', alpha=0.52-0.12*j)
    plt.scatter(x[-1], predict[-1], s=50+5**j, c='red', alpha=0.52-0.12*j)
plt.plot(x, y, c='black', alpha=0.6, label='expected output')
plt.plot(x, predict, c='red', alpha=0.8, linewidth=1., label='real output')
plt.legend(loc='upper left')
plt.savefig('linear03.png')


regresor  = linear_model.LinearRegression()

plt.figure(figsize=(15,10))
plt.suptitle("Using Linear Regression", fontsize="x-large")
for k in range(25):
    y = train['seq'][k+1]
    x = np.linspace(1, len(y),len(y))
    subX = pd.DataFrame({'x':x})
    addpolynomialterms(subX, x)
    plt.subplot(5,5,1+k)
    # regresor
    regresor.fit(subX.as_matrix(), y)
    predict   = regresor.predict(subX.as_matrix())
    # plot the points
    plt.scatter(x, y, s=20, c='green', alpha=0.4)
    # shows the expected and real output
    for j in range(5):
        plt.scatter(x[-1], y[-1], s=10+5**j, c='darkblue', alpha=0.52-0.12*j)
    plt.scatter(x[-1], predict[-1], s=25, c='red')
    plt.plot(x, y, c='gray', alpha=0.3)
    plt.plot(x, predict, c='red', alpha=0.6, linewidth=1.)
    plt.axis('off')
plt.savefig('e02.png')
plt.figure(figsize=(15,10))
plt.suptitle("Test subset", fontsize="x-large")
for k in range(25):
    y = test['seq'][k+1]
    x = np.linspace(1, len(y),len(y))
    plt.subplot(5,5,1+k)
    # plot the points
    plt.scatter(x, y, s=20, c='green', alpha=0.4)
    # shows the expected output
    for j in range(5):
        plt.scatter(x[-1], y[-1], s=10+5**j, c='red', alpha=0.52-0.12*j)
    plt.plot(x, y, c='gray', alpha=0.3)
    plt.axis('off')
plt.savefig('e01.png')
