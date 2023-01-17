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
df = pd.read_csv("/kaggle/input/owid-covid-data.csv", parse_dates = [1,2,3])

df["total_cases"] = df["total_cases"].apply(int)

print(df["location"].unique())
df = df[(df["location"]=="Mexico") & (df["total_cases"]>20)]



def get_date(x):

    return x.day+x.month*31-107



for i in range(3,len(df)):

    df_begining = df.iloc[:i]

    print(i, np.corrcoef(df_begining["date"].apply(get_date), df_begining["total_cases"])[0,1])
def get_date(x):

    return x.day+x.month*31-107



from matplotlib import pyplot as plt



fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.set_yscale('log')



ax.plot(df["date"].apply(get_date), df["total_cases"])

ax.set_yscale('log')



plt.legend()



plt.grid(True)

plt.show()



plt.show()
def get_date(x):

    return x.day+x.month*31-107



from matplotlib import pyplot as plt





fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.set_yscale('log')



ax.plot(df_begining["date"].apply(get_date), df_begining["total_cases"])

ax.set_yscale('log')



plt.legend()



plt.grid(True)

plt.show()



plt.show()
fig = plt.figure()

ax = fig.add_subplot(1,1,1)

#ax.set_yscale('log')



x = range(0, 37)

ax.plot(x, 26*np.exp(chi*x), label = "По модели")



ax.plot(df_begining["date"].apply(get_date), df_begining["total_cases"], label = "Фактически")

ax.set_yscale('log')



plt.legend()



plt.grid(True)

plt.show()



plt.show()
dot(expt, expt)
dot(expt, delta)
fig = plt.figure()

ax = fig.add_subplot(1,1,1)

ax.set_yscale('log')



I_inf = alpha



x = range(0, 37)

ax.plot(x, I_inf/(1+(I_inf/26-1)*np.exp(-chi*x)), label = "По модели")



ax.plot(df["date"].apply(get_date), df["total_cases"], label = "Фактически")

ax.set_yscale('log')



plt.legend()



plt.grid(True)

plt.show()



plt.show()
I_inf
from math import log

from numpy import dot

from math import exp

#ln I = \chi t + ln I_0

for i in range(2,len(df)):

    df_begining = df.iloc[:i]

    chi = dot(df_begining["date"].apply(get_date), df_begining["total_cases"].apply(lambda x: x/26).apply(log))/dot(df_begining["date"].apply(get_date), df_begining["date"].apply(get_date))

    expt = df["date"].apply(get_date).apply(lambda x: exp(chi*x)-1)

    delta = (df["total_cases"].apply(lambda x: 1/x)*df["date"].apply(get_date).apply(lambda x: exp(chi*x))).apply(lambda x: x-1/26)

    alpha = dot(expt, expt)/dot(expt, delta)

    err = df["total_cases"]

    err = err.reset_index()

    I_0=26

    del err["index"]

    x = range(len(err))

    err1 = err.copy()

    err2 = err.copy()

    err1 = pd.concat([err, pd.DataFrame((I_inf/(1+(I_inf/26-1)*np.exp(-chi*x))))], axis=1)

    err2 = pd.concat([err, pd.DataFrame(I_0*np.exp(chi*x))], axis=1)

    print((err1["total_cases"]-err1[0]).apply(lambda x: x*x).sum(),(err2["total_cases"]-err2[0]).apply(lambda x: x*x).sum())
err = df["total_cases"]

err = err.reset_index()

del err["index"]

x = range(len(err))

err = pd.concat([err, pd.DataFrame((I_inf/(1+(I_inf/26-1)*np.exp(-chi*x))))], axis=1)

print((err["total_cases"]-err[0]).apply(lambda x: x*x).sum())
fig = plt.figure()

ax = fig.add_subplot(1,1,1)

I_inf = alpha



x = range(0, 80)

ax.plot(x, I_inf/(1+(I_inf/26-1)*np.exp(-chi*x)), label = "По модели")

ax.plot(df["date"].apply(get_date), df["total_cases"], label = "Фактически")

ax.plot([50]*len(x), I_inf/(1+(I_inf/26-1)*np.exp(-chi*x)), color="black")





plt.legend()



plt.grid(True)

plt.show()

nex = (I_inf/(1+(I_inf/26-1)*np.exp(-chi*x)))[1:]

pre = (I_inf/(1+(I_inf/26-1)*np.exp(-chi*x)))[:-1]     

print(np.argmax((nex-pre)<0.01*pre))