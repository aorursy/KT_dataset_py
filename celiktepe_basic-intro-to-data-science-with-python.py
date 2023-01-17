import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

data = pd.read_csv('../input/athlete_events.csv')
data.info()
data.describe()
data.head()
data.corr()
data.tail()
data.columns
#it brings the year columns from data. kind = it will be line plot, color = line's color

#alpha = line's opacity, label = the name of the line, grid = style of the surface(backgorund)

#linewidth = border of the line

data.Weight.plot(kind='line', color='red', alpha=0.5,linewidth=1, label='Weight', grid=True)

data.Age.plot(color='blue', alpha=0.7,linewidth=1, label='Age', grid=True)

plt.legend(loc='upper right')

plt.xlabel('x axis')  # the name of the label on the bottom

plt.ylabel('y axis')  # the name of the label on the right side

plt.title("Comparison")  # the name of the plot

plt.show() # it shows the plot
#data Scatter

#x = age , y = weight

data.plot(kind='scatter', x = 'Age', y = 'Weight', alpha=0.5, color='red')

plt.xlabel('Age')

plt.ylabel('Weight')

plt.show()
#Histogram

data.Age.plot(kind='hist', bins=50, figsize=(11,8))

plt.show()
data = pd.read_csv('../input/athlete_events.csv')
series = data['Year']  # This is a series definition

print(type(series))

dataFramework = data[['Year']] # This is a Data Framework definition

print(type(dataFramework))
youngAge = data['Age'] < 12 # it brings the list of data which is younger than 12 

data[youngAge]
#this brings the list of data which is younger than 12 and sex is Male

data[(data['Age'] < 12) & (data['Sex'] == 'M')]
for index, value in data[['Age']][0:1].iterrows():

    print(index, " : ", value)

#it brings the age of the first data in the DataFrame
def func():

    nums = [1,2,3]

    return nums

a,b,c = func() # a = nums[0], b = nums[1], c = nums[2]

print(a,b,c)
def merge(name, surname='unnamed'): # surname is already defined, if is doesn't get parameter for surname, surname use defined parameters

    return name + " " + surname



print(merge('Murat'))

print(merge('Murat','Celiktepe'))
def list_args(*args): #it gets parameters that whatever user wants and add them in a list

    for i in args: print(i)

list_args(1,2)

list_args(1,2,3,4)
def dic_args(**kwargs): #like *args , **kwargs gets parameters whatever user wants but add them in a dictionary

    for index, value in kwargs.items():

        print(index + " " + value)

dic_args(name = 'Murat', surname = 'Celiktepe')
square = lambda x: x**2 # x is a parameter and return value is x**2

print(square(4))
nums = [1,2,3]

square = map(lambda x:x**2, nums) # every value in nums list is called in lambda function

print(list(square))
nums = [1,2,3]

nums2 = [i * 2 for i in nums]

print(nums2)
#Example width list comprehension



nums = [10, 20, 30]

nums2 = [i * 4 if i == 20 else i / 2 if i < 11 else i - 10 for i in nums]

print(nums2)
#Another example width List Comprehension



average = sum(data.Year) / len(data.Year)



data['data_Status'] = ["high" if i > average else "low" for i in data.Year]

data.loc[:10,['data_Status','Year']]
data.info()
data.describe()
print(data['Medal'].value_counts(dropna=False))

# (dropna=False) is brings the number of nan values
data.boxplot(column='Weight', by='Sex')
melt_data = data.head()

melt_data
melted_data = pd.melt(frame = melt_data, id_vars = 'Name', value_vars = ['Height', 'Weight'])

melted_data
data1 = data.head()

data2 = data.tail()

conc_data = pd.concat([data1, data2], axis = 0, ignore_index = True) # (ignore_index = True) is ignores ID of the values

conc_data
data.dtypes
data.Year = data.Year.astype('float')

data.head()
data['Medal'].dropna(inplace = True)  

#inplace = means that we do not assign it to new variable. Changes automatically assigned to data

assert data['Medal'].notnull().all()
print(data.Medal.value_counts(dropna=False))

#as we can see, there is no nan values data now because of that we did above
#assert data.Height.notnull().all()

# this code give us an error because, there are nan values
data.Height.fillna('empty', inplace = True)
assert data.Height.notnull().all()

# it did not give error because;

# 'Empty' is written inside of every nan values data 
names = ["John", "Rick"]

ages = ["25", "56"]

column_label = ["Name", "Age"]

list_column = [names, ages]

list1 = list(zip(column_label, list_column))

data_dic = dict(list1)

df = pd.DataFrame(data_dic)

df
#add new columns

df["Team"] = ["Madrid", "Barcelona"]

df
data1 = data.loc[:,["Age","Height","Weight"]]

data1.plot()
data1.plot(subplots = True)

plt.show()
data1.plot(kind="hist", y = "Age", bins = 100, range = (0,60), normed = True)

plt.show()
data1.plot(kind="hist", y = "Age", bins = 100, range = (0,60), normed = True, cumulative=True)

plt.show()
data.head()
import warnings

warnings.filterwarnings("ignore")



new_Data = data.head()

date_list = ["2000-01-10","2000-02-10","2000-03-10","2019-03-15","2019-03-16"]

datatime = pd.to_datetime(date_list)

new_Data["date"] = datatime



new_Data = new_Data.set_index("date")

new_Data
print(new_Data.loc["2019-03-16"])
new_Data.resample('A').mean()
new_Data.resample('A').first().interpolate('linear')