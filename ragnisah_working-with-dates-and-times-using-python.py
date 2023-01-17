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
#pip install DateTime
from datetime import date
from datetime import time
from datetime import datetime
# Get today's date
today =  date.today()
today
# Components of the date
print("Today's year  :", today.year)
print("Today's month :", today.month)
print("Today's day   :", today.day)
# Today's weekday
print("Today's weekday is :", today.weekday())
days = ['mon','tue','wed','thu','fri','sat','sun']
print("Today's weekday is :", days[today.weekday()])
# Date and Time using Now() function
today  = datetime.now()
today
# Get current time
currTime = datetime.time(datetime.now())
currTime
# Formatting date
def main():
    now = datetime.now()
    print("Today's date and time :", now)
    print("Current year :",now.strftime('%Y'))
    print("Current year :",now.strftime('%y'))
    # '%Y/y'-> year, '%a/A'-> weekday,'%b/B'-> month, '%d'-> day of the month 
    print('Current date :', now.strftime('%d %B, %Y (%A)'))
    
if __name__ =='__main__':
    main()