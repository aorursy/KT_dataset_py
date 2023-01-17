# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'berlin', 'norway':'oslo' }

# Print out the keys in europe
print(europe.keys())

# Print out value that belongs to key 'norway'
print(europe['norway'])

# Add italy to europe
europe['italy'] = 'rome'

# Remove australia
del(europe['spain'])

print(europe)
# Dictionary of dictionaries
europe = { 'spain': { 'capital':'madrid', 'population':46.77 },
           'france': { 'capital':'paris', 'population':66.03 },
           'germany': { 'capital':'berlin', 'population':80.62 },
           'norway': { 'capital':'oslo', 'population':5.084 } }


# Print out the capital of France
print(europe['france']['capital'])

# Create sub-dictionary data
data = {'capital':'rome', 'population': 59.83}

# Add data to europe under key 'italy'
europe['italy'] = data

# Print europe
print(europe)
# Pre-defined lists
names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
dr =  [True, False, False, False, True, True, True]
cpc = [809, 731, 588, 18, 200, 70, 45]

# Import pandas as pd
import pandas as pd

# Create dictionary my_dict with three key:value pairs: my_dict
my_dict = {'country':names, 'drives_right':dr, 'cars_per_cap':cpc}

# Build a DataFrame cars from my_dict: cars
cars = pd.DataFrame(my_dict)

# Print cars
print(cars)

# Definition of row_labels
row_labels = ['US', 'AUS', 'JAP', 'IN', 'RU', 'MOR', 'EG']

# Specify row labels of cars
cars.index = row_labels

# Print cars again
print(cars)
#import pandas as pd
import pandas as pd

#import the train.csv data: titanic
titanic = pd.read_csv('../input/train.csv', index_col = 0)

#print out titanic
print(titanic)

import pandas as pd
titanic = pd.read_csv('../input/train.csv', index_col=0)

# Print out Name column as Pandas series
print(titanic['Name'])

# Print out DataFrame with Name and Sex columns
print(titanic[['Name', 'Sex']])

# Print out first 2 observations
print(titanic[0:2])

import pandas as pd
titanic = pd.read_csv('../input/train.csv', index_col = 0)

#Print out passengerID 2's every information
print(titanic.iloc[1])
print(titanic.loc[[2],:])

# Print out passengerID 3's Age
print(titanic.loc[3, 'Age'])
print(titanic.loc[[3],['Age']])

# Print out passengerID 1 and 2's Name, Sex, and Age
print(titanic.loc[[1,2],['Name','Sex','Age']])


# Create arrays
import numpy as np
my_house = np.array([18.0, 20.0, 10.75, 9.50])
your_house = np.array([14.0, 24.0, 14.25, 9.0])

# my_house greater than 18.5 or smaller than 10
print(np.logical_or(my_house>18.5,my_house<10 ))

# Both my_house and your_house smaller than 11
print(np.logical_and(my_house<11, your_house<11))
# Define variables
room = "bed"
area = 14.0

# if-elif-else construct for room
if room == "kit" :
    print("looking around in the kitchen.")
elif room == "bed":
    print("looking around in the bedroom.")
else :
    print("looking around elsewhere.")

# if-elif-else construct for area
if area > 15 :
    print("big place!")
elif area > 10:
    print("medium size, nice!")
else :
    print("pretty small.")
# Import cars data
import pandas as pd
titanic = pd.read_csv('../input/train.csv', index_col = 0)

# select the age column
age = titanic.loc[:,'Age']

# Do comparison on age column
youth = age<30

# Use result to select passengers
print(titanic[age<30]) # print(titanic[youth]) would work the same
# Initialize offset
offset = 8

# Code the while loop
while offset !=0 :
    print("correcting...")
    offset = offset - 1
    print(offset)
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Code the for loop
for a in areas :
    print(a)

# house list of lists
house = [["hallway", 11.25], 
         ["kitchen", 18.0], 
         ["living room", 20.0], 
         ["bedroom", 10.75], 
         ["bathroom", 9.50]]
         
# Build a for loop from scratch
for h in house :
    print("the " + h[0] + " is "+ str(h[1]) + " sqm")
# areas list
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Change for loop to use enumerate()
for index, a in enumerate(areas) :
    print("room " + str(index) + ": " + str(a))
# Definition of dictionary
europe = {'spain':'madrid', 'france':'paris', 'germany':'bonn', 
          'norway':'oslo', 'italy':'rome', 'poland':'warsaw', 'australia':'vienna' }
          
# Iterate over europe
for x, y in europe.items() :
    print("the capital of " + x + " is " + y)
# Import numpy as np
import numpy as np
np_height = [65, 75,65,78,98]
np_student = [[65,78],
              [75,63],
              [65,77],
              [78,86],
              [98,95]]

# For loop over np_height
for height in np_height: 
    print(str(height) + " inches")

# For loop over np_student
for all in np.nditer(np_student): 
    print(all)
