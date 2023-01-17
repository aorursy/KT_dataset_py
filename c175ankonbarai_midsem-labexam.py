# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#1import pandas as pd
data=pd.read_csv("../input/titanic/train_and_test2.csv")
print("Enter roll no.")
value=int(input(""))
data.loc[value-1,['Age','Fare','sibsp']]

#4
class Person:
    def __init__(self, name, city, hobbies):
        self.name = name
        self.city = city
        self.hobbies = hobbies
    def show_info(self):
        print("Name: "+self.name)
        print("Residential City : "+self.city)
        print("Hobbies: ")
        for x in range(len(self.hobbies)):
            print(self.hobbies[x])

hobbies =[ "Playing Football","Riding","Collecting coins"]
p1 = Person("Ankon Barai","Kolkata",hobbies)
p1.show_info()
#2
invalue = 9
inp = int(input('Enter roll no'))
mvalue = int(inp % invalue)
print('Modulo value=', mvalue)