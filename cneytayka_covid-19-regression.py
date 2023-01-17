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
data=pd.read_csv("/kaggle/input/covid19-in-turkey/covid_19_data_tr.csv")
data.head()

data.drop(["Province/State"],axis=1,inplace=True)
data.info()

data.describe()
data["Deaths"].value_counts(dropna=False)
data.head()
#genel bir şeklini inceleyelim.
plt.scatter(data.Deaths,data.Confirmed)
plt.xlabel("Confirmed") # Tanı konulan kişi sayısı
plt.ylabel("Deaths") # Ölen  Kişi sayısı
plt.title("ORAN")
plt.legend()
plt.show()
from sklearn.linear_model import LinearRegression

lr=LinearRegression()
x=data.Confirmed.values.reshape(-1,1)
y=data.Deaths.values.reshape(-1,1)

#fit etme 
lr.fit(x,y)
#kesişim
b0=lr.intercept_
print(b0)
#egim
b1=lr.coef_
print(b1)

y_head=lr.predict(x)
plt.plot(x,y_head,color="red")

plt.show()
data.head()
data.tail()
from sklearn.linear_model import LinearRegression
x1=data.iloc[:,[2,3]].values
y1=data.Confirmed.values.reshape(-1,1)
multi_lr=LinearRegression()
multi_lr.fit(x1,y1)
y_head1=multi_lr.predict(x1)
plt.plot(x1,y_head1)
#kesişim noktası B0
print(multi_lr.intercept_)
#b1 ve b2
print(multi_lr.coef_)