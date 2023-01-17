# Example

print(5 / 8)
# Put code below here

5/8
print(7+10)
# Addition, subtraction

print(5 + 5)

print(5 - 5)
# Multiplication, division, modulo, and exponentiation

print(3 * 5)

print(10 / 2)

print(18 % 7)

print(4 ** 2)
# How much is your $100 worth after 7 years?

print(100*1.1**7)
# Calculate the BMI

height = 1.79

weight = 68.7

#BMI = Weight/Height^2



bmi = weight/height**2

print(bmi)
# TYPE

print(type(bmi))

print(type(5))

print(type("Anwar Iqbal"))

bool = True

print(type(bool))

#sum operator can behave differently

#Different type = Different behaviour

'ab'+'cd'
# Create a variable savings

savings = 100



# Create a variable growth_multiplier

growth_multiplier = 1.1



# Calculate result

result = savings * growth_multiplier ** 7



# Print out result

print(result)
savings = 100

growth_multiplier = 1.1

desc = "compound interest"



# Assign product of growth_multiplier and savings to year1

year1 = savings*growth_multiplier



# Print the type of year1

print(type(year1))





# Assign sum of desc and desc to doubledesc

doubledesc = desc+desc



# Print out doubledesc

print(doubledesc)
# Definition of savings and result

savings = 100

result = 100 * 1.10 ** 7



# Fix the printout

print("I started with $" + str(savings) + " and now have $" + str(result) + ". Awesome!")



# Definition of pi_string

pi_string = "3.1415926"



# Convert pi_string into float: pi_float

pi_float = float(pi_string)
# area variables (in square meters)

hall = 11.25

kit = 18.0

liv = 20.0

bed = 10.75

bath = 9.50



# Create list areas

areas = [hall, kit, liv, bed, bath]



# Print areas

print(areas)

print(type(areas))
# area variables (in square meters)

hall = 11.25

kit = 18.0

liv = 20.0

bed = 10.75

bath = 9.50



# Adapt list areas

areas = ["hallway", hall, "kitchen", kit, "living room", liv, "bedroom", bed, "bathroom", bath]



# Print areas

print(areas)
# area variables (in square meters)

hall = 11.25

kit = 18.0

liv = 20.0

bed = 10.75

bath = 9.50



# house information as list of lists

house = [["hallway", hall],

         ["kitchen", kit],

         ["living room", liv],

         ["bedroom", bed],

         ["bathroom", bath],

         ]



# Print out house

print(house)



# Print out the type of house

print(type(house))
# Create list areas

areas = [hall, kit, liv, bed, bath]



# Print areas index values

print(areas[0])

print(areas[-1])

print(areas[3])
# area variables (in square meters)

hall = 11.25

kit = 18.0

liv = 20.0

bed = 10.75

bath = 9.50



# Adapt list areas

areas = ["hallway", hall, "kitchen", kit, "living room", liv, "bedroom", bed, "bathroom", bath]



# Print areas

print(areas[4:7])

print(areas[:7])

print(areas[4:])
# Create the areas list

areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]



# Sum of kitchen and bedroom area: eat_sleep_area

eat_sleep_area =(areas[3]+areas[7])



# Print the variable eat_sleep_area

print(eat_sleep_area)
# Create the areas list

areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]



# Use slicing to create downstairs

downstairs = areas[:6]



# Use slicing to create upstairs

upstairs = areas[6:]



# Print out downstairs and upstairs

print(downstairs, upstairs)
x = [["a", "b", "c"],

     ["d", "e", "f"],

     ["g", "h", "i"]]

print(x[2][0])

print(x[2][:2])
# Changing elements

# area variables (in square meters)

hall = 11.25

kit = 18.0

liv = 20.0

bed = 10.75

bath = 9.50



# Adapt list areas

areas = ["hallway", hall, "kitchen", kit, "living room", liv, "bedroom", bed, "bathroom", bath]



# Print areas

print(areas)

areas[5]= 15.0

print(areas)

#Adding elements

# area variables (in square meters)

hall = 11.25

kit = 18.0



# Adapt list areas

areas = ["hallway", hall, "kitchen", kit]



# Print areas

print(areas)



areas_ext = areas + ["Living", 25]

print(areas_ext)
#Removing elements

# area variables (in square meters)

hall = 11.25

kit = 18.0

liv = 20.0

bed = 10.75

bath = 9.50



# Adapt list areas

areas = ["hallway", hall, "kitchen", kit, "living room", liv, "bedroom", bed, "bathroom", bath]



# Print areas

print(areas)



del(areas[5])

print(areas)
#Behind the scene = assigned a reference not the actual value

x = ["a", "b", "c"]

y = x

print(x)

print(y)

y[1] = "z"

y

print(x)

print(y)





#Use list function to assign lists 

x = ["a", "b", "c"]

y = list(x)

print(x)

print(y)

# or y = x[:]

y[1] = "z"

print(x)

print(y)
# Create the areas list

areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]



print(areas)

# Correct the bathroom area

areas[-1]=10.50



# Change "living room" to "chill zone"

areas[4]="chill zone"

print(areas)
# Create the areas list and make some changes

areas = ["hallway", 11.25, "kitchen", 18.0, "chill zone", 20.0,

         "bedroom", 10.75, "bathroom", 10.50]



print(areas)

# Add poolhouse data to areas, new list is areas_1

areas_1=areas+["poolhouse", 24.5]

print(areas_1)

# Add garage data to areas_1, new list is areas_2

areas_2=areas_1+["garage",15.45]

print(areas_2)
x = ["a", "b", "c", "d"]

del(x[1])

print(x)
# Create list areas

areas = [11.25, 18.0, 20.0, 10.75, 9.50]



# Create areas_copy

areas_copy = areas[:]



# Change areas_copy

areas_copy[0] = 5.0



# Print areas

print(areas)
# Create list areas

areas = [11.25, 18.0, 20.0, 10.75, 9.50]

Maximum = max(areas)

print(Maximum)
round(1.68, 1)
#Closest integer without second arguement

round(1.68)
# Create variables var1 and var2

var1 = [1, 2, 3, 4]

var2 = True



# Print out type of var1

print(type(var1))



# Print out length of var1

print(len(var1))



# Convert var2 to an integer: out2

out2=int(var2)
# Create lists first and second

first = [11.25, 18.0, 20.0]

second = [10.75, 9.50]



# Paste together first and second: full

full = first + second





# Sort full in descending order: full_sorted

full_sorted = sorted(full, reverse=True)



# Print out full_sorted

print(full_sorted)
# Create the areas list and make some changes

#Index method on list and string

areas = ["hallway", 11.25, "kitchen", 18.0, "chill zone", 20.0,

         "bedroom", 10.75, "bathroom", 10.50]

print(areas.index("kitchen"))

print(areas.count(10.75))

anwar = "anwar"

print(anwar.index("r"))

print(anwar.capitalize())

print(anwar.replace("w", "i45"))

# Create the areas list and make some changes

#Index method on list and string

areas = ["hallway", 11.25, "kitchen", 18.0, "chill zone", 20.0,

         "bedroom", 10.75, "bathroom", 10.50]

print(areas)

areas.append("CP")

print(areas)



areas.extend("ko098")

print(areas)
# string to experiment with: place

place = "poolhouse"



# Use upper() on place: place_up

place_up = place.upper()



# Print out place and place_up

print(place, place_up)



# Print out the number of o's in place

print(place.count("o"))
# Create list areas

areas = [11.25, 18.0, 20.0, 10.75, 9.50]



# Print out the index of the element 20.0

print(areas.index(20.0))



# Print out how often 9.50 appears in areas

print(areas.count(9.50))
# Create list areas

areas = [11.25, 18.0, 20.0, 10.75, 9.50]



# Use append twice to add poolhouse and garage size

areas.append(24.5)

areas.append(15.45)





# Print out areas

print(areas)





# Reverse the orders of the elements in areas

areas.reverse()



# Print out areas

print(areas)
import numpy as np

np.array([1, 2, 3])



#OR



from numpy import array

array([1, 2, 3])
# Definition of radius

r = 0.43



# Import the math package

import math



# Calculate C

C = 0

C = 2* math.pi * r

# Calculate A

A = 0

A = math.pi*r**2

# Build printout

print("Circumference: " + str(C))

print("Area: " + str(A))
# Definition of radius

r = 192500



# Import radians function of math package

from math import radians



# Travel distance of Moon over 12 degrees. Store in dist.

phi = radians(12)

dist = r * phi



# Print out dist

print(dist)