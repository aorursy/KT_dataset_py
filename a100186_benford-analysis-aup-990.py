import seaborn as sns

import matplotlib.pyplot as plt

import math

import csv

import sys

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("../input/raw-benford-numbers-edited/Raw Benford Numbers.csv", index_col = 'Unnamed: 0')
#an example of some of our data

df.tail()
## Remove all "nan" values for blank cells

all = []

for x in df:

    all.append(df[x].values)



cData = []

for i in range(0,3):

    cleanedList = [x for x in all[i] if str(x) != 'nan']

    cData.append(cleanedList)

finalData = []

for z in range(len(cData)):

    for y in cData[z]:

        finalData.append(y)

#finalData is the cleaned data without nan values, we still have to clear the trailing decimal points and zeroes

#this for and if loop evaluates if the value ends with '.0' and if it does, the last two digits are removed

for val in range(len(finalData)):

    if str(finalData[val])[-2:] == '.0':

        finalData[val] = str(finalData[val])[:-2]
print(finalData)
def firstDigit(input_list):

    myList = [0,0,0,0,0,0,0,0,0]

    for num in input_list:

        output = str(num)[:1]

        myList[int(output)-1] += 1

    return myList



def secondDigit(input_list):

    myList = [0,0,0,0,0,0,0,0,0,0]

    for num in input_list:

        output = str(num)[1:2]

        myList[int(output)] += 1

    return myList



def thirdDigit(input_list):

    myList = [0,0,0,0,0,0,0,0,0,0]

    for num in input_list:

        output = str(num)[2:3]

        myList[int(output)] += 1

    return myList

second_Digit = secondDigit(finalData)

third_Digit = thirdDigit(finalData)

first_Digit = firstDigit(finalData)
Benford_percentiles = pd.DataFrame({

    'First Digit Expected': [0, .301, .176, .125, .097, .079, .067, .058, .051, .046],

    'Second Digit Expected': [.12, .114, .109, .104, .100, .097, .093, .090, .088, .085],

    'Third Digit Expected': [.102, .101, .101, .101, .100, .100, .099, .099, .099, .098]

                                    })
Benford_percentiles
#This is just a quick script to seperate out the expected first digit from the rest of the Data Frame as these are on a

#scale of 1-9 instead of 0-9 like the second and third digits

First_digit_benfords = []

for x in Benford_percentiles['First Digit Expected']:

    if x > 0:

        First_digit_benfords.append(x)
first_digit_percentile = [(x / sum(first_Digit)) for x in first_Digit]

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

index = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

x = np.arange(len(index))

plt.xticks(x, index)

ax.plot(x, First_digit_benfords, label= 'Actual')

ax.plot(x, first_digit_percentile, label= 'Expected')

plt.title('First Digit; Expected values are in blue, actual are in orange')

plt.show
second_Digit_percentile = [(x / sum(second_Digit)) for x in second_Digit]

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

x = np.arange(len(index))

plt.xticks(x, index)

ax.plot(x, Benford_percentiles['Second Digit Expected'])

ax.plot(x, second_Digit_percentile)

plt.title('Second Digit; Expected values are in blue, actual are in orange')

plt.show()
third_Digit_percentile = [(x / sum(third_Digit)) for x in third_Digit]

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

x = np.arange(len(index))

plt.xticks(x, index)

ax.plot(x, Benford_percentiles['Third Digit Expected'])

ax.plot(x, third_Digit_percentile)

plt.title('Third Digit; Expected values are in blue, actual are in orange')

plt.show()
#reinitialize these variables before I alter them just to keep the code clean

first_digit_percentile = [(x / sum(first_Digit)) for x in first_Digit]

first_digit_percentile.insert(0, 0)

index = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']



final_series = {'First Digit Test': first_digit_percentile,

                'Second Digit Test': second_Digit_percentile,

                'Third Digit Test': third_Digit_percentile}

digit_df = pd.DataFrame(data=final_series, index=index)

final_df = pd.merge(digit_df, Benford_percentiles, on=digit_df.index, how='outer')
final_df
final_df.plot(kind='box', rot=-30)
print('Skew\n', final_df.skew(), '\nKurtosis:\n', final_df.kurt())
final_df.std()
def myDataFrame(sample, digit_test, expected):

    difference = []

    for x in range(len(digit_test)):

        difference.append(digit_test[x]-expected[x])

    if len(sample) < 10:

        sample.insert(0, 0)

    output = pd.DataFrame({

        'Sample Count': sample, 

        'Digit Test (%)': digit_test, 

        'Expected Values (%)': expected, 

        'Difference (%)': difference

                        })

    return output
#This runs the function we created before on each of my different tests

first_digit_df = myDataFrame(first_Digit, final_df['First Digit Test'], final_df['First Digit Expected'])

second_digit_df = myDataFrame(second_Digit, final_df['Second Digit Test'], final_df['Second Digit Expected'])

third_digit_df = myDataFrame(third_Digit, final_df['Third Digit Test'], final_df['Third Digit Expected'])



#now I convert the numbers into percentage values to make my data tables more readable

for x in [first_digit_df, second_digit_df, third_digit_df]:

    for y in ['Digit Test (%)', 'Expected Values (%)', 'Difference (%)']:

        x[y] = round(x[y].apply(lambda i: i*100), 2)



#Below is an example of the completed data table

first_digit_df
def last_two_digit_test(input_list):

    '''accepts a dataframe as an input and returns a list of the count of each digit out of 100 digits'''

    #create an list of 100 integers, each with a zero value

    myList = []

    for _ in range(100):

        myList.append(0)

    for num in input_list:

        num = str(num)

        output = str(num)[-2:]

        myList[(int(output))-1] += 1

    return myList
last_two = last_two_digit_test(finalData)
index = []

for x in range(100):

    index.append(x)



#Plotting

fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

width = .35

x = np.arange(len(index))

plt.xticks(x, index)

ax.bar(x - width/2, last_two, width= width)

plt.title('Last Two Digit Count')

plt.show()
last_two_df = pd.DataFrame(last_two, index=index)

print([ j for (i,j) in zip(last_two, index) if i >= 4 ])
#We create new variables called 'significant numbers' which we will alter in order to deliver us just the significant digits



first_digit_significant_numbers = first_digit_df.copy()

second_digit_significant_numbers = second_digit_df.copy()

third_digit_significant_numbers = third_digit_df.copy()



#This for loop adds the significant digit to a list called 'temp' when the difference of the number is greater than 5%

temp = []

for x in [first_digit_significant_numbers, second_digit_significant_numbers, third_digit_significant_numbers]:

    temp.append(x.index.where(x['Difference (%)'] > 5).dropna())



#Now we convert the list into a cleaner version which will be easier to work with and put it in a dictionary

significant_numbers = {

    'First Digit Significant Values' : temp[0].astype(int).values,

    'Second Digit Significant Values' : temp[1].astype(int).values,

    'Third Digit Significant Values' : temp[2].astype(int).values

}

#here is an example of what those numbers are

significant_numbers
#This cell changes all of my integers into strings which are easier to slice in order to locate the sections of the 990 that

#might contain fraudulent data

finalData = [str(x) for x in finalData]
#these lines below actually are finding the numbers in our entire dataset where the digit at some given position

#matches the number and the position that we are looking for

first_numbers = [x for x in finalData if x[:1] in significant_numbers['First Digit Significant Values'].astype(str)]

second_numbers = [x for x in finalData if x[1:2] in significant_numbers['Second Digit Significant Values'].astype(str)]

third_numbers = [x for x in finalData if x[2:3] in significant_numbers['Third Digit Significant Values'].astype(str)]
print(first_numbers, second_numbers, third_numbers)
def intersection(lst1, lst2): 

    lst3 = [value for value in lst1 if value in lst2] 

    return lst3 
inter_one_two = intersection(first_numbers, second_numbers)

inter_one_three = intersection(first_numbers, third_numbers)

inter_two_three = intersection(second_numbers, third_numbers)

interfinal = intersection(inter_one_two, third_numbers)

print(inter_one_two)

print(inter_one_three)

print(inter_two_three)

print(interfinal)
in_first = set(inter_one_two)

in_second = set(inter_two_three)



in_second_but_not_in_first = in_second - in_first



result = inter_one_two + list(in_second_but_not_in_first)
print(result)
#delete all values that arent on our list 'result'

all_to_investigate = df[df.isin(result)]



#remove every column that is entirely 'NaN' values

all_to_investigate = all_to_investigate.dropna(axis='columns', how='all')



#remove every row that is entirely 'NaN' values

all_to_investigate = all_to_investigate.dropna(axis='index', how='all')
all_to_investigate
to_drop = ['revenue less expenses', 'Total expenses']

final_list = all_to_investigate.drop(to_drop)

final_list
final_list