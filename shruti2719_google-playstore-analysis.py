import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
data=pd.read_csv("../input/googleplaystore.csv")
data.head()
data.tail()
data.info()
data.describe()
data=data.dropna()
data.info()
data.nunique()
plt.figure(figsize=(12,12))

sns.countplot(y='Category',data=data)

plt.title("Number of Apps available based on Category")
x=data['Rating']
ax=sns.kdeplot(x)
x.mean()
az=data['Type'].value_counts()

az
l=['Free','Paid']

plt.pie(az,labels=l,autopct='%1.1f%%',radius=1.5)

plt.savefig('pie.jpg')



plt.axis('equal')

plt.show()
w=data['Price']
data['Price'] = data['Price'].str.replace('.', '')

data['Price'] = data['Price'].str.replace('$', '')

data['Price'] = data['Price'].astype(int)
w=data['Price']
w.mean()


plt.scatter(w,x)

plt.legend()

plt.show()