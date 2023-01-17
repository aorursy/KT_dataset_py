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
from datetime import date
print("What's your year of birth? [Ex: 1995]")
Year=input(Year)
print("What's your month of birth? [Ex: 12]")
Month=input(Month)
print("What's your day of birth? [Ex: 30]")
Date=input(Date)
print('Your DOB is',(Date+"/"+Month+"/"+Year))
today_day=date.today()
today_day
age=today_day.year-int(Year)
print('You are',age,'years old')
if((int(Month)==12 and int(Date)>=22) or (int(Month)==1 and int(Date)<=19)):
     sign=("\n Capricorn")
elif((int(Month)==1 and int(Date)>=20) or (int(Month)==2 and int(Date)<=18)):
     sign=("\n Aquarius")
elif((int(Month)==2 and int(Day)>=19) or (int(Month)==3 and int(Day)<=20)):
     sign=("\n Pisces")
elif((int(Month)==3 and int(Day)>=21) or (int(Month)==4 and int(Day)<=19)):
     sign=("\n Arie")
elif((int(Month)==4 and int(Day)>=20) or (int(Month)==5 and int(Day)<=20)):
     sign=("\n Taurus")
elif((int(Month)==5 and int(Day)>=21) or (int(Month)==6 and int(Day)<=20)):
     sign=("\n Gemini")
elif((int(Month)==6 and int(Day)>=21) or (int(Month)==7 and int(Day)<=22)):
     sign=("\n Cancer")
elif((int(Month)==7 and int(Day)>=23) or (int(Month)==8 and int(Day)<=22)):
     sign=("\n Leo")
elif((int(Month)==8 and int(Day)>=23) or (int(Month)==9 and int(Day)<=22)):
     sign=("\n Virgo")
elif((int(Month)==9 and int(Day)>=23) or (int(Month)==10 and int(Day)<=22)):
     sign=("\n Libra")
elif((int(Month)==10 and int(Day)>=23) or (int(Month)==11 and int(Day)<=21)):
     sign=("\n Scorpio")
elif((int(Month)==11 and int(Day)>=22) or (int(Month)==12 and int(Day)<=21)):
     sign=("\n Sagittarius")
print("Your Zodiac Sign is",sign)