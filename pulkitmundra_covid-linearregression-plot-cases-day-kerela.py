import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
df=pd.read_csv('../input/covid19-in-india/covid_19_india.csv')

df
df=df.drop("ConfirmedIndianNational",axis=1)

df=df.drop("ConfirmedForeignNational",axis=1)

df
df.keys()
new_data = df[df["State/UnionTerritory"] == "Kerala"]

new_data
new_data['Date'] = pd.to_datetime(new_data.Date, format='%d/%m/%y').astype(str)
new_data[0:15]
new_data['Date']=new_data['Date'].str.replace("-","")
new_data['Date'][0:10]
x=new_data["Date"]

y=new_data["Confirmed"]
x=new_data.iloc[:,0:1].values
lr=LinearRegression()

lr.fit(x,y)

# m = lr.coef_

# c = lr.intercept_
y_predict = lr.predict(x)

y_predict
plt.figure(figsize=(20,10))

plt.scatter(x,y)

plt.plot(x, y_predict, color="red")

plt.xlabel("Date")

plt.ylabel("Number of cases")

plt.show()