# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df = pd.read_csv("/kaggle/input/main-data-set-seo/Main Data File v.2 (csv).csv",encoding= 'unicode_escape')
df.head(20)

condition_1 = str(input("Enter the campaign season: 1. All year, 2. Summers, 3.Winters: "))

if condition_1  == "All year":
    abcd = df.loc[(df['Seasons ']== "All year ")].copy()
elif condition_1 == "Summers":
    abcd = df.loc[(df['Seasons ']== "Summers")].copy()
elif condition_1 == "Winters":
    abcd = df.loc[(df['Seasons ']== "Winters")].copy()
else:
    print("Please type in correctly!")

condition_2 = str(input('Enter target country: France, USA, Germany, UK, Australia: '))

if condition_2 == 'France':
    abcd = abcd.loc[(abcd['Location']== "France")].copy()
elif condition_2 == 'USA':
    abcd = abcd.loc[(abcd['Location']== "USA")].copy()
elif condition_2 == 'Germany':
    abcd = abcd.loc[(abcd['Location']== "Germany")].copy()
elif condition_2 == 'UK':
    abcd = abcd.loc[(abcd['Location']== "UK")].copy()
elif condition_2 == 'Australia':
    abcd = abcd.loc[(abcd['Location']== "Australia")].copy()
else:
    print("Please type in correctly!")

condition_3 = str(input('Enter mentioned Customer Category: Hiking, Winter Sports, Vacations/Casual Travellers: '))

if condition_3 == 'Hiking':
    abcd = abcd.loc[(abcd['Customer Category']== "Hiking")].copy()
elif condition_3 == 'Winter Sports':
    abcd = abcd.loc[(abcd['Customer Category']== "Winter Sports")].copy()
elif condition_3 == 'Vacations/Casual Travellers':
    abcd = abcd.loc[(abcd['Customer Category']== "Vacations/Casual Travellers")].copy()
else:
    print("Please type in correctly!")
if abcd.empty is True:
    print("No keywords in these categories. Try again!!!")
else:
    condition_3 = int(input('No of months for the campaign: '))
    abcd['Cost in $']= abcd['Top of page bid (high range) [Budget]']*condition_3
    abcd['Estimated Reach'] = abcd['Estimated Reach'].str.replace(',', '')
    abcd['Estimated Reach'] = abcd['Estimated Reach'].astype(int)
    abcd['Cost to reach'] = (abcd['Cost in $']/abcd['Estimated Reach'])*100
    print(abcd[['Keyword','Estimated Reach','Cost in $','Cost to reach','Competition']].head(10))
    condition_4 = str(input("Select a column(Estimated Reach,Cost in $,Cost to reach) to filter: "))
    condition_5 = int(input("Type: 1 for Ascending or Type: 2 for Descending: "))
    if condition_5 == 0:
        print(abcd[['Keyword','Estimated Reach','Cost in $','Cost to reach','Competition']].sort_values(condition_4,ascending=0))
    else:
        print(abcd[['Keyword','Estimated Reach','Cost in $','Cost to reach','Competition']].sort_values(condition_4,ascending=1))
        #fig = plt.figure(figsize=(14,15))
        #sns.countplot(x=abcd['KEI (Keyword Effectiveness Score)'], data=abcd)

