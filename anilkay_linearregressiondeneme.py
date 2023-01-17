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
import random

random.randint(0,10000)
x1_list=[]

x2_list=[]

y_list=[]

#3x1+2x2=y

i=0

while i<=150:

    x1=random.randint(0,10000)

    x2=random.randint(0,10000)

    y=3*x1+2*x2

    x1_list.append(x1)

    x2_list.append(x2)

    y_list.append(y)

    i=i+1
data=pd.DataFrame({

    "x1":x1_list,

    "x2":x2_list,

    "y":y_list

})
data.head()
x=data[["x1","x2"]]

y=data["y"]

from sklearn.linear_model import LinearRegression

linear=LinearRegression()

linear.fit(x,y)
linear.score(x,y)
linear.predict([[3,4]])
i=0

toplamhata=0

while i<=150:

    x1=random.randint(0,10000)

    x2=random.randint(0,10000)

    y=3*x1+2*x2

    sonuc=int(linear.predict([[x1,x2]]))

    print(str(y)+"   "+str(sonuc))

    toplamhata=toplamhata+abs(y-sonuc)

    i=i+1

print("Sum of Error: "+str(toplamhata))    
np.array(linear.coef_).astype(float)
linear.coef_[1]
linear.coef_[0]
x1_list=[]

x2_list=[]

y_list=[]

#3x1+2x2=y

i=0

while i<=150:

    x1=random.randint(0,10000)

    x2=random.randint(0,10000)

    y=3*x1+2*x2+44

    x1_list.append(x1)

    x2_list.append(x2)

    y_list.append(y)

    i=i+1

data=pd.DataFrame({

    "x1":x1_list,

    "x2":x2_list,

    "y":y_list

})

x=data[["x1","x2"]]

y=data["y"]

linear=LinearRegression()

linear.fit(x,y)

linear.predict([[3,4]])
print(np.array(linear.coef_).astype(float)[0])

print(np.array(linear.coef_).astype(float)[1])
i=0

toplamhata=0

while i<=150:

    x1=random.randint(0,10000)

    x2=random.randint(0,10000)

    y=3*x1+2*x2+44

    sonuc=int(linear.predict([[x1,x2]]))

    print(str(y)+"   "+str(sonuc))

    toplamhata=toplamhata+abs(y-sonuc)

    i=i+1

print("Sum of Error: "+str(toplamhata))  