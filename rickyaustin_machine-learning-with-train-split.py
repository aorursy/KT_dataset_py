# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

%matplotlib inline

sns.set()



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

df.head()
newTable = df[['hotel','lead_time','adults','deposit_type','customer_type']].copy()

newTable.head()
newTable.info()
g = sns.catplot(x='hotel', col='customer_type', kind='count', data=newTable);

g.fig.set_figwidth(10)

g.fig.set_figheight(10)
print(newTable.customer_type.value_counts())

print(newTable.deposit_type.value_counts())

print(newTable.hotel.value_counts())
cust = {'Transient': 1,'Transient-Party': 2,'Contract':3,'Group':4}

depo = {'No Deposit': 1,'Non Refund': 2,'Refundable':3} 

hot = {'City Hotel': 1,'Resort Hotel': 2}



newTable.customer_type = [cust[item] for item in newTable.customer_type]

newTable.deposit_type = [depo[item] for item in newTable.deposit_type]

newTable.hotel = [hot[item] for item in newTable.hotel]



newTable.describe()

sns.stripplot(x='hotel', y='deposit_type', data=newTable, alpha=0.3, jitter=True);
g = sns.PairGrid(newTable, hue="hotel")

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)

g.add_legend();
y = newTable.hotel

x = newTable.drop('hotel',axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

x_train.head()
model = LinearRegression().fit(x_train,y_train)

predictions = model.predict(x_test)

plt.scatter(y_test,predictions)

plt.xlabel('Nilai asli')

plt.ylabel('Prediksi')

plt.show()
knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(x_train, y_train)

print(knn.score(x_test, y_test))