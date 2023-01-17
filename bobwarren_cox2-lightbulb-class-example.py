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
class Lightbulb(object):
    
    #Constructor Method
    def __init__(self, status):
        self.status = status
        

    #Another Method
    def Lightbulb_ON(self):
        self.status=True
       
    
    def Lightbulb_OFF(self):
        self.status=False
    
    def Get_status(self):
        return self.status
    
    def toggle(self):
        if self.status:
            self.status=False
        else:
            self.status=True
            
    
    
    
bulb=Lightbulb(False)
print(bulb.Get_status())
bulb.Lightbulb_ON()

print(bulb.Get_status())

bulb.toggle()
print (bulb.Get_status())

bulb.toggle()
print (bulb.Get_status())

