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
def answer(s):
    characters = 'abcdefghijklmnopqrstuvwxyz'
    string = ''
    for t in s:
        if characters.find(t) != -1:
            i = characters.find(t)
            i += 1
            string = string + characters[-i]
        else: 
            string = string + t
    return string
            
    # your code here
print(answer('I olev krv!!@#'))
def answer(x, y):
    x0 = 1
    dia = 1
        
    while x0 < x + y - 1 :
        dia = dia + x0
        x0 += 1
        #print(dia)
    
    pho = dia + x - 1
    return pho
        
    # your code here
print(answer(5,6))