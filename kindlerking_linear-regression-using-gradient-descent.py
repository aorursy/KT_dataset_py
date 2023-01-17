# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("../input/years-of-experience-and-salary-dataset/Salary_Data.csv")
#ease of data
df['Salary(in 1000s)']=df['Salary']/1000
df.drop(columns={'Salary'})
#Seperating X and Y
X=df.iloc[ : ,0]
Y=df.iloc[ : ,1]
#vizualising nature of data
plt.scatter(X,Y)
plt.show()
#Building the model
m=0
c=0
L=0.0001
epochs=1000
n=float(len(X))

for i in range(epochs):
    Y_curr=(m*X)+c
    m_grad=(-2/n)*sum((X*Y-X*Y_curr))   
    c_grad=(-2/n)*sum((Y-Y_curr))
    m=m-(L*m_grad)
    c=c-(L*c_grad)

print('m is:',m,'c is:',c)   
#The algorithm is very useful in learning the basic working principle of Linear Regression. 
#m and c are constantly updated with a  ccuracy of 0.0001 as our learning rate and it goes on 1000 times to give us a good result. 
    
Y_Pred= m*X+c
plt.scatter(X,Y)
plt.plot([min(X),max(X)],[min(Y_Pred),max(Y_Pred)],color='red')
plt.show()
