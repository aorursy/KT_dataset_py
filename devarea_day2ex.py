ls = [[1,2],[3,4]]
ls[1][0]
for i in range(2):

    for j in range(2):

        ls[i][j]*=10
import numpy as np
arr = np.array([[1,2],[3,4]])
arr
arr*=10
arr
arr > 20
arr[arr>20] = 25
arr
help(arr)
arr1 = np.arange(1,20,2)
arr1
ls = [1,2,3,4,5,6,7]
ls2 = ls[1:6:2]
arr2 = arr1[1:6:2].copy()
ls2
ls2[0]=400
ls
arr2
arr2[1]= 400
import matplotlib.pyplot as plt
x = np.array([1,2,3,4,5])
y = np.array([11,34,27,100,76])
plt.plot(x,y)
plt.scatter(x,y)
plt.bar(x,y)
pic = plt.imread('logo1.png')
plt.imshow(pic)
type(pic)
pic.shape
pic[:32,32:,2] = 255 # [r,g,b,a] 0-255

pic[:32,32:,1] = 0

pic[:32,32:,0] = 0
import pandas as pd

import seaborn as sb
df = pd.read_csv('../input/pupils.csv')
df.head()
df.info()
df.describe()
df.Age-=10
df.Country.unique()
df.Country.nunique()
df.Country.value_counts()
sb.pairplot(df,hue='gen')
sb.distplot(df.Age)
df.corr()['Weight']['Height']
df.corr()
#%%timeit

df[(df.Age > 10) & (df.income > 30000)]
#%%timeit

df.query("Age > 10 and income > 30000")
df.groupby(['Country']).mean()
df.groupby(['Country','gen']).sum()
df.head()
df.drop(1,inplace=True)
df.drop(['Name','Country'],axis=1,inplace=True)
df['n1'] = df.income / df.family
df.head()
df['namelen'] = df.Name.apply(len) 
def op1(v):

    print("====",v)

    if v > 40000:

        return 1

    elif v >20000:

        return 2

    else:

        return 3
df['newinc1'] = df.income.apply(op1)
arr = df.Age.values
arr
arr*=10
df = pd.read_csv('../input/pupils.csv')
df.head()
df['n1'] = df.income / df.family
import sklearn.linear_model as sl
X = df[['Age','Height','Weight','income','rooms','family','type','n1']]
y = df.avg
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=101)
model = sl.LinearRegression()

#model = sl.ARDRegression()
model.fit(X_train,y_train)
pred = model.predict(X_test)
model.score(X_test,y_test)
sb.distplot(y_test - pred)
import sklearn.metrics as mt
mt.mean_absolute_error(y_test,pred)
mt.r2_score(y_test,pred)
np.sqrt(mt.mean_squared_error(y_test,pred))
pred
y_test
model.predict([[7,100,30,20000,4,3,1,6666]])
model.coef_
model.intercept_
import seaborn as sb
df = sb.load_dataset('diamonds')