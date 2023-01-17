# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline



from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_turkey = pd.read_csv("/kaggle/input/Korona.csv",sep =";")
df_turkey.head()
df_turkey.info()
df_turkey.rename(columns={'Total Case': 'Total_Case', 'Total Death': 'Total_Death'}, inplace=True)
df_turkey.head()
df_turkey["Tests"]=["-","-","-","-","-","-","-","2015","1981","3656","2952","1738","3672","3952","5035","7286","7533","7641","9982","11535","15422","14397","18757","16160","19664","20065","21400","20023","24900","28578","30864","33170","35720"]

df_turkey.Tests.replace(["-"],0.0,inplace=True)





plt.figure(figsize= (15,10))

sns.barplot(x=df_turkey.Tarih, y=df_turkey.Total_Case)

plt.xticks(rotation=45)

plt.xlabel("Date")

plt.ylabel("Number of Total Case")

plt.title("Total Cases")
data = go.Scatter(

                    x = df_turkey.Tarih,

                    y = df_turkey.Case,

                    mode = "lines",

                    name = "Cases",

                    marker = dict(color = 'rgba(25, 130, 5, 0.8)'),

                    text= df_turkey.Total_Case)

layout = dict(title = 'Cases Daily',

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
data = go.Scatter(

                    x = df_turkey.Tarih,

                    y = df_turkey.Tests,

                    mode = "lines",

                    name = "Cases",

                    marker = dict(color = 'rgba(25, 130, 5, 0.8)'),

                    text= df_turkey.Case)

layout = dict(title = 'Number of Test',

              xaxis= dict(title= 'Date',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
a=df_turkey.iloc[7:,5]

b=df_turkey.iloc[7:,1] 

data =go.Scatter(

                    x = a,

                    y = b,

                    mode = "markers",

                    marker = dict(color = 'rgba(240, 0, 220, 0.8)'),

                    text= df_turkey.Tarih)

layout = dict(title = 'Tests vs Daily Cases',

              xaxis= dict(title= 'Tarih',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Daily Cases',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
plt.figure(figsize= (15,10))

sns.barplot(x=df_turkey.Tarih, y=df_turkey.Death)

plt.xticks(rotation=45)

plt.xlabel("Date")

plt.ylabel("Death")

plt.title("Death")
a = df_turkey.iloc[10:23,1]

b = df_turkey.iloc[20:33,3]

c = df_turkey.iloc[20:33,0]

data = go.Scatter(

                    x = a,

                    y = b,

                    mode = "markers",

                    name = "Cases",

                    marker = dict(color = 'rgba(25, 130, 5, 0.8)'),

                    text= c

                    )

layout = dict(title = 'Number of Death(Starting from 01.04.2020) vs Ten Days Before Daily Case',

              xaxis= dict(title= 'Daily Case from 21.03.2020',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Daily Death from 01.04.2020',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
plt.figure(figsize= (15,10))

sns.barplot(x=df_turkey.Tarih, y=df_turkey.Total_Death)

plt.xticks(rotation=45)

plt.xlabel("Date")

plt.ylabel("Total Death")

plt.title("Total Death")
trace1 = go.Box(

    y=df_turkey.Total_Case,

    name = 'Total Case',

    marker = dict(

        color = 'rgb(12, 128, 128)',

    )

)

data = [trace1]

iplot(data)
b=[]

for i in df_turkey.index:

    a = [i + 1]

    b.extend(a)

df_turkey["day_number"]=b
x = df_turkey.day_number.values.reshape(-1,1)



y=df_turkey.Total_Case.values.reshape(-1,1)



from sklearn.preprocessing import PolynomialFeatures



poly_reg = PolynomialFeatures(degree=4)

x_poly = poly_reg.fit_transform(x)



from sklearn.linear_model import LinearRegression

linear_reg2 = LinearRegression()



linear_reg2.fit(x_poly,y)

plt.scatter(x,y)



plt.xlabel("Day Number")

plt.ylabel("Total Cases")

y_head = linear_reg2.predict(x_poly)





plt.plot(x,y_head,color="green")

plt.show()



from sklearn.metrics import r2_score



print("r_score",r2_score(y,y_head))
x_ = np.arange(0, 80, 1).reshape(-1,1)

x_poly2 = poly_reg.fit_transform(x_)

y_head2 = linear_reg2.predict(x_poly2)



plt.plot(x_,y_head2,color="green")

plt.show()

print("13.04.2020 prediction = ",y_head2[34])

x = df_turkey.day_number.values.reshape(-1,1)



y=df_turkey.Total_Case.values.reshape(-1,1)



from sklearn.preprocessing import PolynomialFeatures



poly_reg3 = PolynomialFeatures(degree=5)

x_poly3 = poly_reg3.fit_transform(x)



from sklearn.linear_model import LinearRegression

linear_reg3 = LinearRegression()



linear_reg3.fit(x_poly3,y)

plt.scatter(x,y)



plt.xlabel("Day Number")

plt.ylabel("Total Cases")

y_head3 = linear_reg3.predict(x_poly3)





plt.plot(x,y_head3,color="green")

plt.show()



from sklearn.metrics import r2_score



print("r_score",r2_score(y,y_head3))
x_3 = np.arange(0, 80, 1).reshape(-1,1)

x_poly3 = poly_reg3.fit_transform(x_3)

y_head3 = linear_reg3.predict(x_poly3)





plt.plot(x_3,y_head3,color="green")

plt.show()



print("13.04.2020 prediction = ",y_head3[34])
a = df_turkey.iloc[10:23,1].values.reshape(-1,1)

b = df_turkey.iloc[20:33,3].values.reshape(-1,1)

c = a/b

c=c.reshape(-1,1)
from sklearn.preprocessing import PolynomialFeatures



poly_reg4 = PolynomialFeatures(degree=2)

x_poly4 = poly_reg4.fit_transform(a)



from sklearn.linear_model import LinearRegression

linear_reg4 = LinearRegression()



linear_reg4.fit(x_poly4,c)

plt.scatter(a,c)



plt.xlabel("Case from 21.03.2020")

plt.ylabel("Case/Death from 01.04.2020")

a_3 = np.arange(0, 3500, 1).reshape(-1,1)

x_poly5 = poly_reg4.fit_transform(a_3)

y_head5 = linear_reg4.predict(x_poly5)

plt.plot(a_3,y_head5,color="green")

plt.show()

from sklearn.metrics import r2_score

y_head4 = linear_reg4.predict(x_poly4)

print("r_score",r2_score(c,y_head4))
a2 = df_turkey.iloc[24:34,1].values.reshape(-1,1)

x_predict = poly_reg4.fit_transform(a2)

y_predict = linear_reg4.predict(x_predict)

y_predict = a2/y_predict





plt.scatter(a2,y_predict,color = "Red")
