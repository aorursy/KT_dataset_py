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
import numpy as np

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

actual=[1,1,0,1,0,0,1,0,0,0]

predicted=np.random.rand(10)

print(predicted)

for i in range(10):

    if(predicted[i]<0.5):

        predicted[i]=0

    else:

        predicted[i]=1

print(predicted)

results=confusion_matrix(actual.predicted)

print('confusion matrix:')

print(results)

print('Accuracy score:'.accuracy_score(actual.predicted))

print('report:')

print(classification_report(actual,predicted))
import pandas as pd

import matplotlib.pyplot as plt



iris = pd.read_csv("../input/iris/Iris.csv")



df = iris[['SepalLengthCm','Species']]



df.boxplot(by='Species', column=['SepalLengthCm'], grid=False)



spc = ['Iris-virginica']



df1 = df.loc[df['Species'].isin(spc)]

#print(df1)



df2 = df1.loc[df1['SepalLengthCm'] <= 5.0]['SepalLengthCm']



print(df2)

df1.loc[106,'SepalLengthCm'] = 10.0





print(df1)
import matplotlib.pyplot as plt

import pandas as pd

Iris = pd.read_csv("../input/iris/Iris.csv")

Iris=pd.read_csv("../input/iris/Iris.csv")

fig=plt.gcf()

fig.set_size_inches(10,7)

fig=sns.boxplot(x='Species',y='PetalLengthCm',data=Iris,order=['Iris-virginica','Iris-versicolor','Iris-setosa'],linewidth=2.5,orient='v',dodge=False)
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

Iris = pd.read_csv("../input/iris/Iris.csv")





sns.scatterplot(Iris['SepalLengthCm'],Iris['SepalWidthCm'],data=Iris, hue="Species")

plt.xlabel('Sepal length')

plt.ylabel('Sepal width')

plt.title('Scatter plot on Iris dataset')
import numpy as np

from sklearn.model_selection import KFold

data=np.random.rand(10)

kfold = KFold(5,True,1)

for train,test in kfold.split(data):

    print('train:%s,test:%s'%(data[train],data[test]))
from sklearn.linear_model import LogisticRegression

from sklearn import datasets

import numpy as np



%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt



iris = datasets.load_iris()



print(list(iris.keys()))



#print(iris)



X = iris["data"][:,3:]  # petal width

y = (iris["target"]==2).astype(np.int)



log_reg = LogisticRegression(penalty="l2")

log_reg.fit(X,y)



X_new = np.linspace(0,3,1000).reshape(-1,1)

y_proba = log_reg.predict_proba(X_new)



plt.plot(X,y,"b.")

plt.plot(X_new,y_proba[:,1],"g-",label="Iris-Virginica")

plt.plot(X_new,y_proba[:,0],"b--",label="Not Iris-Virginca")

plt.xlabel("Petal width", fontsize=14)

plt.ylabel("Probability", fontsize=14)

plt.legend(loc="upper left", fontsize=14)

plt.show()



log_reg.predict([[1.7],[1.5]])
iris = datasets.load_iris()



X = iris["data"][:,(2,3)]  # petal length, petal width

y = iris["target"]



softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=5)

softmax_reg.fit(X,y)



X_new = np.linspace(0,3,1000).reshape(-1,2)



plt.plot(X[:, 0][y==1], X[:, 1][y==1], "y.", label="Iris-Versicolor")

plt.plot(X[:, 0][y==0], X[:, 1][y==0], "b.", label="Iris-Setosa")



plt.legend(loc="upper left", fontsize=14)



plt.show()
