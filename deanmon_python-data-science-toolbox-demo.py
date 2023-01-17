# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd 

data = pd.read_csv('../input/Pokemon.csv')

def tuple_ex():

    t = (data.Speed,data.Attack,data.Defense)

    return t

a, b, c = tuple_ex()

print(a,b,c)
x = data.head().HP

def f():

    x = data.head().Speed

    return x

print(x)

print(f())

x = data.head().Attack

def f():

    y = 2*x

print(x)

print(f())
def square():

    def add():

        x = data.head().HP

        y = data.head().Defense

        z = x+y

        return z

    return add()**2

print(square())
def f(*args):

    for i in args:

        print(i)

f(data.head().HP)

print("")

f(data.head().HP,data.head().Attack,data.head().Defense)

def f(**kwargs):

    for key, value in kwargs.items():

        print(key, " ", value)

f(Tür = data.head().Name,Can = data.head().HP)
name = "Pikachu"

it = iter(name)

print(next(it))

print(*it)
threshold = sum(data.head().Speed)/len(data.head().Speed)

print("Threshold", threshold)

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]]
