# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
mean=0
median=0
mode=0
a=[]
b=[]

data = pd.read_csv('../input/ka.csv')
mean=data['min_price'].mean()
median=data['min_price'].median()
a=data['min_price'] 
b=data['district']
print("average minimum arecanut price per quintal Rs",mean)
print("median minimum arecanut price per quintal Rs",median)
#print("",mode)
try:
    print("modular minimum arecanut price per quintal Rs",statistics.mode(a))
except statistics.StatisticsError :
    print("mode does not exist")



def sct():
	plt.scatter(a, b, color='g')
	plt.xlabel('price')
	plt.ylabel('district')
	plt.show()
def barh():
	plt.barh(b, a, color='r')
	plt.xlabel('price')
	plt.ylabel('district')
	plt.title("District wise  minimum price distribution of Arecanut in Karnataka ")
	plt.show()
sct()
barh()

import pandas as pd
import statistics
import matplotlib.pyplot as plt
import numpy as np
a=[]
data = pd.read_csv('../input/ka.csv')
a=data['min_price']
plt.boxplot(a)
plt.ylabel('price')
plt.title('Arecanut price distribution throught state of Karnataka')
plt.show()
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
np.set_printoptions(threshold=np.nan)
from sklearn.metrics import mean_squared_error 
 
dataset = pd.read_csv('../input/ka.csv')
space=dataset['min_price']
price=dataset['max_price']

x = np.array(space).reshape(-1, 1)
y = np.array(price)


#Splitting the data into Train and Test
from sklearn.cross_validation import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=2/3)


#Fitting simple linear regression to the Training Set
from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit(xtrain, ytrain)

#Predicting the prices
pred = regressor.predict(xtest)
#a = mean_squared_error(ytest, ytrain)
#print("Mean squared error:",a )

#Visualizing the training Test Results 
plt.scatter(xtrain, ytrain, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title ("Visuals for Training Dataset of Arecanut in Min vs Max Price")
plt.xlabel("min_price")
plt.ylabel("max_price")
plt.show()

#Visualizing the Test Results 
plt.scatter(xtest, ytest, color= 'red')
plt.plot(xtrain, regressor.predict(xtrain), color = 'blue')
plt.title("Visuals for Test DataSet of Arecanut in Min vs Max Price")
plt.xlabel("min_price")
plt.ylabel("max_price")
plt.show()