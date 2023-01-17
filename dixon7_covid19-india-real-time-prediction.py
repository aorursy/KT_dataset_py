# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
from datetime import datetime,timedelta
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import sys
import requests
import re
from bs4 import BeautifulSoup

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
country ='India'
if(country == "India"):
    website_url = requests.get("https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_India").text
    soup = BeautifulSoup(website_url,'html.parser')
    #print(soup.prettify())
    link = soup.find("div",{"class":"barbox tright"})
    table1 = link.find('table').find_all('tr',{"class":"mw-collapsible"})
    data=[]
    for item in table1:
        if item.findAll('td')[0].text!='⋮':
            data.append([item.findAll('td')[0].text,int(re.sub(r'\(.*\)', '',item.findAll('td')[2].text.replace('\n','')).replace(',','')),item.findAll('td')[3].text])

    df = pd.DataFrame(data,columns=['Date','total_cases','new'])
    #df = pd.read_csv(".\IndiaCases.csv")
    df = df.loc[:,['Date','total_cases']]
    FMT ='%Y-%m-%d'
    date = df['Date']
    df['Date'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01", FMT)).days  )


if(country == "Italy"):
    url = "https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv"
    df = pd.read_csv(url)
    df = df.loc[:,['data','totale_casi']]
    FMT = '%Y-%m-%dT%H:%M:%S'
    date = df['data']
    df['data'] = date.map(lambda x : (datetime.strptime(x, FMT) - datetime.strptime("2020-01-01T00:00:00", FMT)).days  )


def logistic_model(x,a,b,c):
    return c/(1+np.exp(-(x-b)/a))
x = list(df.iloc[:,0])
y = list(df.iloc[:,1])
df
fit = curve_fit(logistic_model,x,y,p0=[10,150,500000],maxfev=10000)
#a= fit[0][0]
#b= fit[0][1]
#c= fit[0][2]
maxval = fit[0]
a,b,c =maxval
print("Best case Prediction")
print("Rate:",int(a), "Number of days:",int(b)+28, "Prediction:",int(c) )
print("worst case")
a1,b1,c1 = np.amax(fit[1], axis=0)
print("Rate:",int(a1), "Number of days:",int(b1)+28, "Prediction:",int(c1) )

errors = [np.sqrt(fit[1][i][i]) for i in [0,1,2]]
sol = int(fsolve(lambda x : logistic_model(x,a,b,c) - int(c),b))
def exponential_model(x,a,b,c):
    return a*np.exp(b*(x-c))

exp_fit = curve_fit(exponential_model,x,y,p0=[1,1,1], maxfev=10000)


pred_x = list(range(max(x),sol))
plt.rcParams['figure.figsize'] = [7, 7]

plt.rc('font', size=14)
# Real data
plt.scatter(x,y,label="Real data",color="red")
# Predicted logistic curve
plt.plot(x+pred_x, [logistic_model(i,fit[0][0],fit[0][1],fit[0][2]) for i in x+pred_x], label="Logistic model" )
# Predicted exponential curve
plt.plot(x+pred_x, [exponential_model(i,exp_fit[0][0],exp_fit[0][1],exp_fit[0][2]) for i in x+pred_x], label="Exponential model" )
plt.legend()
plt.xlabel("Days since 1 January 2020")
plt.ylabel("Total number of infected people")
plt.ylim((min(y)*0.9,c*1.1))

plt.show()