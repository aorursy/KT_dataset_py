# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy.optimize import curve_fit



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
def plot_and_predict(more_days, country):    

    def avg_err(pcov):

        return np.round(np.sqrt(np.diag(pcov)).mean(), 2)

    

    cmd = data[data["country"]==country].iloc[: , [0, 2, 3 ,4, 5]].copy() 

    cmd_grp = cmd.groupby("date")[['confirm', 'death', 'recover']].sum().reset_index()

    y = cmd_grp["confirm"]

    x = np.arange(len(y))

    

    print(np.array(y).astype(int))



    def f_lin(x, a, b):

        return a * x + b



    def f_poly(x, a, b, c, d, e):

        return a * x**4 + b*x**3 + c*x**2 + d*x**1 + e



    def f_pow(x, a, b, c):

        return b*(x)**a + c

        

    def f_exp(x, a, b, c):

        return a * np.exp(-b * x) + c

    

    def f_sigmoid(x, a, b, c, d):

        return c / (1 + np.exp(-b*(x-a)))+d





    

    # popt_lin, pcov_lin = curve_fit(f_lin, x, y)

    # popt_poly, pcov_poly = curve_fit(f_poly, x, y)

    popt_pow, pcov_pow = curve_fit(f_pow, x, y,maxfev=100000)

    popt_exp, pcov_exp = curve_fit(f_exp, x, y, p0=(1, 1e-6, 1), maxfev=100000)

    popt_sig, pcov_sig = curve_fit(f_sigmoid,x, y, method='dogbox', bounds=([10., 0.001, y.mean(), 10],[100, 1., 10*y.mean(), 100]), maxfev=200000)



    ## Prediction

    plt.figure(figsize=(18,12))

    x_m = np.arange(len(y)+more_days)

    

    # y_m = f_lin(x_m, *popt_lin)

    # plt.plot(x_m, y_m, c='k', marker="*", label="linear | error: "+str(avg_err(pcov_lin))) 

    # plt.text(x_m[-1]+.5, y_m[-1], str(int(y_m[-1])), size = 10)

    

    # y_m = f_poly(x_m, *popt_poly)    

    # plt.plot(x_m, y_m, c='m', marker="+", label="polynomial | error: "+str(avg_err(pcov_poly))) 

    # plt.text(x_m[-1]+.5, y_m[-1], str(int(y_m[-1])), size = 10)

    

    y_m = f_exp(x_m, *popt_exp)

    plt.plot(x_m, y_m, c='r', marker="p", label="UGLY (exponential) | error: "+str(avg_err(pcov_exp))) 

    plt.text(x_m[-1]+.5, y_m[-1], str(int(y_m[-1])), size = 15, color="r")



    y_m = f_pow(x_m, *popt_pow)

    plt.plot(x_m, y_m, c='y', marker="s", label="BAD (power law) | error: "+str(avg_err(pcov_pow))) 

    plt.text(x_m[-1]+.5, y_m[-1], str(int(y_m[-1])), size = 15, color="y")

       

    

    y_m = f_sigmoid(x_m, *popt_sig)

    plt.plot(x_m, y_m, c='g', marker="x", label="GOOD (sigmoid) | error: "+str(avg_err(pcov_sig))) 

    plt.text(x_m[-1]+.5, y_m[-1], str(int(y_m[-1])), size = 15, color="g")

    

    y = y.values

    plt.plot(x, y, c='b', marker="o", label = "Official Data")

    plt.text(x[-1]-2.5, y[-1], str(int(y[-1])), size = 15, color="b")



    plt.xlabel("Days", size=14)

    plt.xticks(np.arange(1,len(x_m),2),size=14)

    plt.ylabel("Total Infected", size=14)

    plt.yticks(size=14)

    plt.legend(prop={'size': 15})

    plt.title(country+"'s Data", size=15)

    plt.axvline(x[-1], color="b")

    plt.text(x[-1]-1, 1, "today", color="b", size = 20, rotation=90)

    plt.axvline(x[-1]+1, color="k")

    plt.text(x[-1]+1, 1000, "tomorrow", color="k", size = 20, rotation=90)



    plt.show()
data=pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data=data.drop('Last Update', axis=1)

data=data.drop("SNo",axis=1)

data=data.rename(columns={"ObservationDate": "date", "Country/Region": "country", "Province/State": "state","Confirmed":"confirm","Deaths": "death","Recovered":"recover"})

data.head()
next_days = 1
country = "UK"

plot_and_predict(next_days, country)
country = "US"

plot_and_predict(next_days, country)
country = "South Korea"

plot_and_predict(next_days, country)
country = "Italy"

plot_and_predict(next_days, country)
country = "Iran"

plot_and_predict(next_days, country)