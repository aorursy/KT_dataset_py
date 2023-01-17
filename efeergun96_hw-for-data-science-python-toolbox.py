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
# Getting the data from directory to the Notebook.

salaries_by_region = pd.read_csv("../input/salaries-by-region.csv") 

salaries_by_college_type = pd.read_csv("../input/salaries-by-college-type.csv")

degrees_pay_back = pd.read_csv("../input/degrees-that-pay-back.csv")
# exploring the data types and Columns more

print("-------")
print("Salaries by Region")
print("-------")
print(salaries_by_region.info())

print("-------")
print("Salaries by College Type")
print("-------")
print(salaries_by_college_type.info())

print("-------")
print("Degrees that pay Back")
print("-------")
print(degrees_pay_back.info())

#first thing first we should define the problem by seeing te Data.

salaries_by_region.head(10)
def fix_my_salary(x):   #let's assume x is "$70,400.00"
    """this is docstring"""
    x = str(x);   # making it string to work easily.
    x = x[1:]     # this deletes the first char, which leaves us with x = "70,400.00"
    x = x.replace(",",".") # this replaces the " , " with math friendly " . "  x = "70.400.00"
    x = x[:-3]    # this would remove the unnecesarry cents and would make data simpler.  x = "70.400"
    x = float(x)  # this converts our data type to float  x = 70.4  (thousand dollars.)
    return x
    
example = salaries_by_region['Starting Median Salary'][4]  #getting any example data
print(" input : " + example)
example = fix_my_salary(example)
print(" output : " + str(example))  # looks good.
x = 0;
for i in salaries_by_region['Starting Median Salary']:
    salaries_by_region['Starting Median Salary'][x] = fix_my_salary(salaries_by_region['Starting Median Salary'][x])
    x = x + 1;
salaries_by_region.head(8)
salaries_by_region['Mid-Career 75th Percentile Salary'] = list(map(fix_my_salary, salaries_by_region['Mid-Career 75th Percentile Salary']))
salaries_by_region['Mid-Career Median Salary'] = list(map(fix_my_salary, salaries_by_region['Mid-Career Median Salary']))
salaries_by_region['Mid-Career 25th Percentile Salary'] = list(map(fix_my_salary, salaries_by_region['Mid-Career 25th Percentile Salary']))
salaries_by_region.head(8)
def high_low_salary(salary):
    """this function creates 2 lists from one big list.
    first array is higher values
    second array is lower values
    """
    high = list(salary > np.median(salary));   # getting the all values that are higher than median as True
    low =  list(salary < np.median(salary));   # getting the all values that are lower than median as True
    
    high = np.array(salary[high])  # creating an array of only high scores
    low = np.array(salary[low])    # creating an array of only low scores
    
    
    return high,low;


high,low = high_low_salary(salaries_by_region['Starting Median Salary'])   # can be used for any type of plot that shows diference between low and high
print(high)
print("---")     # testing my function
print(low)

import matplotlib.pyplot as plt

plt.clf()
plt.figure(figsize=(10,10))
plt.plot(high[:100], c='r')
plt.plot(low[:100], c='b')
plt.ylabel("Salary")
plt.show()                       # there is clearly no sense of doing this plot for the topic, but as mentioned on the top reason is increasing my python skills.
plus_10_percent = lambda x: x + x*0.1;

increased_salary = list(map(plus_10_percent,salaries_by_region['Starting Median Salary']))
#To see the difference I will zip original and new salaries on one array


zipped = zip((salaries_by_region['Starting Median Salary']),increased_salary)

zipped = list(zipped)

print(zipped[7])
print(zipped[17])
#Example of Default value on a function

degrees_pay_back.head(10)
def good_inreaser(increase=25):
    """this function prints the jobs those get good increases depended on given increase value.
    the increase value setted to 25 as default
    """
    x = list(degrees_pay_back['Percent change from Starting to Mid-Career Salary'] > increase)
    print(list(degrees_pay_back['Undergraduate Major'][x]))
    
good_inreaser();   # default example increase = 25%
good_inreaser(increase=90)  # example with given value  increase = 90%


