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
class Computer:

    def _init_(self):

        print("I am a Computer")

        def config (self):

            print ("i5,16gb")

com1=Computer()

class Computer:

    def __init__(self):

        print("I am a Computer")

    def __call__(self, a):

        print(a+10)

    def __str__(self):

        return "hello"

    #def config (self):

        #print ("i5,16gb")

com1=Computer()

print(com1)

com1(5)
class Computer:

    def __init__(self,cpu,ram):

        self.cpu=cpu

        self.ram=ram

        print("I am a Computer")

    def config (self):

        print ("config is", self.cpu,self.ram)

com1=Computer("i5","16gb")

com1.cpu="Ryzen 5"

com1.config()

print (id(com1))



com2=Computer("ryzen 3","8gb")

com2.config()
