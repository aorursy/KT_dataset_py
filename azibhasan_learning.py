# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/salary-data/Salary_Data.csv")
df
df.shape
df.size
df.info()
df['fruit_name'].unique()
df['fruit_name'].nunique()
df['fruit_subtype'].unique()
df1=pd.read_csv("../input/fruits/fruits_data.csv")
df1
import numpy as np
print(np.__version__)
a=np.array(10)
a
type(a)
print(a.size)
print(a.ndim) #0th dimensional array is called a a scalar

b=np.array([1,2,3])
b
print(b.size)
print(b.ndim) #1 D array is called a vector
print(b.shape)
c=np.array([[1,2,3],[4,5,6]])
print(c)
print(c.size)
print(c.ndim)
print(c.shape)
d=np.random.randint(0,10,10)
d
d.reshape(2,5)
d.sort()
d
np.append(d,9)
np.power(d,2)
x=np.ones([10])
x
y=np.zeros([9])
y
xys=np.arange(0,12,2)
xys
np.linspace(0,11,6)
import matplotlib.pyplot as plt
x=[1,2,3,5]
y=[8,3,7,9]
#plt.scatter(x,y,c='#fdb147')
plt.bar(x,y,width=0.5, color=('r','g','k','b'))
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('My line graph')
plt.show()
x=np.linspace(0,10,6)
y=np.zeros(6)
plt.plot(x,y, marker='.')
print(x)
print(y)
overs=[0,4,8,12,16,20]
team1=[0,30,67,98,123,145]
team2=[0,27,75,103,110,137]
plt.plot(overs,team1,marker="*",c='r',label='Team1')
plt.plot(overs,team2,marker="+",c='g',label='Team2')
plt.legend()
plt.xlabel('Overs')
plt.ylabel('Runs')
plt.title('Match Analysis')
plt.show()
import pandas as pd
df=pd.read_csv("../input/salary-data/Salary_Data.csv")
df
df.shape
import matplotlib.pyplot as plt
plt.scatter(df['YearsExperience'],df['Salary'])
plt.show()
x=df.iloc[:,[0]].values
y=df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
x_train.shape
x_test.shape
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred
y_test
df1=pd.DataFrame({'Actual_Salary':y_test,'Predicted_Salary':y_pred,'Difference in %':((y_pred-y_test)/(y_test)*100)})
df1
plt.scatter(x_train,y_train,c='r')
plt.plot(x_train,model.predict(x_train))
plt.scatter(x_test,y_test,c='r')
plt.plot(x_test,model.predict(x_test))

df1.plot(kind='bar')
y_pred_unique=model.predict([[3]])
y_pred_unique
c=model.intercept_
m=model.coef_
y_f=m*3+c
y_f
import joblib
joblib.dump(model,'Salary_Model')
mj=joblib.load('Salary_Model')
mj.predict([[3]])
import numpy as np
xyz=np.random.uniform(1,10,10)
xyz.sort()
xyz
d={'Years':xyz}
d
df3=pd.DataFrame(d)
df3
op=mj.predict(df3)
op
df3['Pred_Salary']=op
df3
df3.to_csv('Prediction.csv',index=False)
pd.read_csv('Prediction.csv')
