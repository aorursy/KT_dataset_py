import numpy as np

from sklearn.model_selection import KFold

data=np.random.rand(10) 

kfold=KFold(5,True,1)

for train,test in kfold.split(data):

        print('train: %s, test: %s' % (data[train], data[test]))
from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

import numpy as np

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

print("confusion matrix")

print(results)

print("accuracy score:".accuracy_score(actual.predicted))

print("report:")

print(classification_report(actual.predicted))
from random import seed

from random import random

seed(1)

for _ in range(10):

    value = random()

    print(value)
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
import matplotlib.pyplot as plt

Iris = pd.read_csv("../input/iris/Iris.csv")

fig=plt.gcf()

fig.set_size_inches(10,7)

fig=sns.boxplot(x='Species',y='SepalLengthCm',data=Iris,order=['Iris-virginica','Iris-versicolor','Iris-setosa'],linewidth=2.5,orient='v',dodge=False) 
import matplotlib.pyplot as plt

import pandas as pd

iris = pd.read_csv("../input/iris/Iris.csv")

df=iris[['SepalLengthCm','Species']]

df.boxplot(by='Species',column=['SepalLengthCm'],grid=False)

spc=['Iris-virginica']

df1=df.loc[df['Species'].isin(spc)]

print(df1)

df2=df1.loc[df1['SepalLengthCm']<=5.0]['SepalLengthCm']

print(df2)

df1.loc[106,'SepalLengthCm']=10.0

print(df1)
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

Iris = pd.read_csv("../input/iris/Iris.csv")

plt.figure(figsize = (10, 7)) 

data=Iris

x = data["SepalLengthCm"]   

plt.hist(x, bins = 20, color = "green") 

plt.title("Sepal Length in cm") 

plt.xlabel("Sepal_Length_cm") 

plt.ylabel("Count") 



import pandas as pd

Iris = pd.read_csv("../input/iris/Iris.csv")
import matplotlib.pyplot as plt

import seaborn as sns

import pandas as pd

Iris = pd.read_csv("../input/iris/Iris.csv")



sns.scatterplot(Iris['SepalLengthCm'],Iris['SepalWidthCm'],data=Iris,hue="Species",palette=['purple','yellow','black'])

plt.xlabel('Sepal length')

plt.ylabel('Sepal Width')

plt.title('Scatter plot on Iris dataset')

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

import statsmodels.api as sm



Iris = pd.read_csv("../input/iris/Iris.csv")

Iris.head()

plt.figure(figsize=(16, 8))



X = Iris['SepalLengthCm'].values.reshape(-1,1)

y = Iris['PetalLengthCm'].values.reshape(-1,1)

reg = LinearRegression()

reg.fit(X, y)

print("The linear model is: Y = {:.5} + {:.5}X".format(reg.intercept_[0], reg.coef_[0][0]))



predictions = reg.predict(X)

plt.figure(figsize=(16, 8))

plt.scatter(

    Iris['SepalLengthCm'],

    Iris['PetalLengthCm'],

    c='black'

)

plt.plot(

    Iris['SepalLengthCm'],

    predictions,

    c='blue',

    linewidth=2

)

plt.xlabel("SepalLength")

plt.ylabel("PetalLength")

plt.show()



X = Iris['SepalLengthCm']

y = Iris['PetalLengthCm']

X2 = sm.add_constant(X)

est = sm.OLS(y, X2)

est2 = est.fit()

print(est2.summary())