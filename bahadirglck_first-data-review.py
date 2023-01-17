# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#import os
#print(os.listdir("../input"))

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/bitflyerJPY_1-min_data_2017-07-04_to_2018-06-27.csv')
data.info()
data.describe()
data.columns
data.head()
data.corr()
#Line Plot >> Line plot is better when x axis is time
data.Timestamp.plot(kind='line', color='g', label='Timestamp', linewidth = 2 ,alpha = 0.8,grid = True, linestyle = '-' )
data.Weighted_Price.plot( kind = 'line', color = 'r', label = 'Weighted_Price', linewidth = 2, alpha = 0.8,grid = True, linestyle = '-.')
plt.legend(loc='upper right')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.show()
#Scatter >> Scatter is better when there is correlation between two variables
data.plot(kind = 'scatter', x = 'Volume_(BTC)', y = 'Volume_(Currency)', alpha = 0.5, color='r')

plt.xlabel('Volume_(BTC)')
plt.ylabel('Volume_(Currency)')
plt.title('Scatter Plot Between V_BTC & V_Curr')
plt.show()
#Histogram >> is better when we need to see distribution of numerical data
#data.plot(kind='hist', x='Weighted_Price', color = 'b', bins= 50, figsize=(7,7))
data.Weighted_Price.plot(kind = 'hist', bins = 50, figsize=(7,7))
plt.show()
plt.clf()
dictionary = {"Ankara" : "METU","Istanbul":"Bogazici University","Bursa": "Uludag University"}
for i in dictionary.keys():
    print(dictionary[i])

#print(dictionary.keys(), dictionary.values())
dictionary["Sakarya"] = "Sakarya Unıversıty"
print(dictionary)
dictionary.clear()
print(dictionary)
data.columns
series = data['Open']
print(type(series))
data_frame = data[['Open']]
print(type(data_frame))
print(False or False)
x = data['Open']<295000
#print(data[x])
print(len(data[x]))
data.Open[-1:]   
data[np.logical_and(data['Open']>500000, data['Close']>500000 )]

lst = "Bahadir"
for i in lst:
    print(i)
print('*******')
for index, value in enumerate(lst):
    print(index, ":",value)
print('*********')
for index, value in data[['Open']][0:2].iterrows():
    print(index, ":",value)
def  tuble_ex():
    """ returned defined t tuble"""
    t = (3,4,5,6)
    return t 
a,b,c,_  = tuble_ex()
print(a,b,c)
def car_name_brand (a, b = " A180", c = "Mercedes"):
    lst = a + " " + b + " " + c 
    return lst
print(car_name_brand("10"))
print(car_name_brand("10","Jetta", "Wolkswagen"))
def f(*args):
    for i in args:
        print(i)
f(1)
f(1,2,3,4)
def f(**kwargs):
    for key, value in kwargs.items():      
        print(key, " ", value)
f(country = 'spain', capital = 'madrid', population = 123456)
sm = lambda x,y : x+y
print(sm(3,4))
name = "NeuralNetworks"
sname = "AI"
y = map(lambda x:x*2,name)
print(list(y))

it = iter(name)
print(*it)

z = zip(name,sname)
print(list(z))

def data_sum(a):
    sm = [ i+i for i in data[a]]
    return sm
print(sum(data_sum("Open"))/2)
print(sum(data.Open))

