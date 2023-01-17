S = [x**2 for x in range(10)]

V = [2**i for i in range(13)]

M = [x for x in S if x % 2 == 0]

print(S)

print(V)

print(M)
numbers = range(30)

new_list = []

for n in numbers:

    if n%2==0: 

        new_list.append(n**2) #raise that element to the power of 2 and append to the list

print(new_list)



new__list = [n**2 for n in numbers if n%2==0] #expression followed by for loop followed by the conditional clause

print(new__list)



kilometer = [39.2, 36.5, 37.3, 37.8]



feet = map(lambda x: float(3280.8399)*x, kilometer)

print(list(feet))



feet_ = [float(3280.8399)*x for x in kilometer]

print(feet_)
def adder(x,y,z):

    print("sum:",x+y+z)



adder(10,12,13)
def adder(*num):

    sum = 0

    for n in num:

        sum = sum + n

    print("Sum:",sum)

adder(3,5)

adder(4,5,6,7)

adder(1,2,3,5,6)
def intro(**data):

    print("\nData type of argument:",type(data))

    for key, value in data.items():

        print("{} is {}".format(key,value))

        

intro(Firstname="Sita", Lastname="Sharma", Age=22, Phone=1234567890)

intro(Firstname="John", Lastname="Wood", Email="johnwood@nomail.com", Country="Wakanda", Age=25, Phone=9876543210)
import numpy as np

import pandas as pd 

import os

import matplotlib.pyplot as plt

import seaborn as sns



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#UTF-8 is a variable-width character encoding standard 

#that uses between one and four eight-bit bytes to represent all valid Unicode code points.



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.       
data = pd.read_csv('/kaggle/input/world-university-rankings/school_and_country_table.csv')

data = pd.read_csv('/kaggle/input/world-university-rankings/timesData.csv')

data = pd.read_csv('/kaggle/input/world-university-rankings/cwurData.csv')

#/kaggle/input/world-university-rankings/education_expenditure_supplementary_data.csv

#/kaggle/input/world-university-rankings/educational_attainment_supplementary_data.csv

#/kaggle/input/world-university-rankings/shanghaiData.csv

data.head(11)
data.info()
data.corr()

# Corrrelation Map 

#The statistical relationship between two variables is referred to as their correlation. 

#A correlation could be positive, meaning both variables move in the same direction, or negative, meaning that 

#when one variable's value increases, the other variables' values decrease.
data.world_rank.plot(kind = 'line', color = 'g',label = 'world_rank',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')

data.national_rank.plot(color = 'b',label = 'national_rank',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')



data.plot(kind='scatter', x='quality_of_education', y='quality_of_faculty',alpha = 0.5,color = 'red')

plt.xlabel('quality_of_education')             

plt.ylabel('quality_of_faculty')

plt.title('quality_of_education - quality_of_faculty Scatter Plot')            
# Histogram

# bins = number of bar in figure

data.publications.plot(kind = 'hist',bins = 50,figsize = (5,5))

plt.show()