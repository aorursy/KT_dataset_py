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
#This function appends numrric values to a list and when you enter a nonnumeric value it ends.
def number_or_not ():
    y = input('Please enter a numeric value: ')
    x = list(y)
    for i in x:
        if i in ('-','+','.','0','1','2','3','4','5','6','7','8','9'):
            continue
        else:
            return print('Input is not a number!')
        break
    if y.count('.') < 2 and  y.count('-') < 2 and  y.count('+') < 2:
        if y.count('-') == 1 and y.index('-') == 0 and y.count('+') == 0:
            if y.count('.') == 1:
                y = float(y)
            else:
                y = int(y)
        elif y.count('+') == 1 and y.index('+') == 0 and y.count('-') == 0:
            if y.count('.') == 1:
                y = float(y)
            else:
                y = int(y)
        elif y.count('-') == 0 and y.count('+') == 0:
            if y.count('.') == 1:
                y = float(y)
            else:
                if y.strip('0')=='' and y!='':
                    y=0
                elif y=='':
                    return print('Input is not a number!')
                else:
                    y=int(y)
        else:
            return print('Input is not a number!')
        return y
    else:
        return print('Input is not a number!')
list_of_numbers=[]
def add_to_list():
    print('\n \n This function appends numrric values to a list and when you enter a nonnumeric value it ends.\n')
    sayi = number_or_not()
    while sayi==0 or sayi:
        list_of_numbers.append(sayi)
        sayi = number_or_not()
    return list_of_numbers
add_to_list()
print(list_of_numbers)


