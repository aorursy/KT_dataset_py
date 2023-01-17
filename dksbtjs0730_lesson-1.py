# Create a variable savings
savings = 100

# Create a variable factor
factor = 1.10

# Calculate result. ** is an operator for exponentiation.
result = savings * factor ** 7

# Print out result
print(result)
type("Hello")
type(3.5)
type(True)
# Several variables to experiment with
savings = 100
desc = "compound interest"

# Assign sum of savings and savings to year1
year1 = savings + savings

# Print the type of year1
print (year1)

# Assign sum of desc and desc to doubledesc
doubledesc = desc + desc

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
# area variables 
hall = 11.25
kit = 18.0
liv = 20.0
bed = 10.75
bath = 9.50

# Create list areas
areas = [hall, kit, liv, bed, bath]

# Print areas
print(areas)

# house information as list of lists
house = [["hallway", hall],
         ["kitchen", kit],
         ["living room", liv],
         ["bedroom", bed],
         ["bathroom", bath]]

# Print out house
print(house)

# Print out the type of house
print(type(house))
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Print out second element from areas
print(areas[1])

# Print out the area of the living room
print(areas[5])

# Print out last element from areas
print(areas[-1])

# You can select multiple elements from your list. Use the following syntax:
# my_list[start:end]
# The start index will be included, while the end index is not.
# print out third and fourth elements from areas
print(areas[2:4])

# It's also possible not to specify these indexes. 
# If you don't specify the begin index, Python figures out that you want to start your slice at the beginning of your list. 
# If you don't specify the end index, the slice will go all the way to the last element of your list
# Use slicing to create downstairs, that contains the first 6 elements of areas.
downstairs = areas[:6]

# Use slicing to create upstairs, that contains the last 4 elements of areas.
upstairs = areas[-4:]

# Print out downstairs and upstairs
print(downstairs)
print(upstairs)
x = [["a", "b", "c"],
     ["d", "e", "f"],
     ["g", "h", "i"]]
print(x[2][0])
print(x[2][:2])
# Create the areas list
areas = ["hallway", 11.25, "kitchen", 18.0, "living room", 20.0, "bedroom", 10.75, "bathroom", 9.50]

# Correct the bathroom area
areas[-1] = 10.50

# Add poolhouse data to areas, new list is areas_1
areas_1 = areas + ["poolhouse", 24.5]

# delete "kitchen" and kitchen area 
del(areas[2:4])

print(areas_1)
print(areas)
x = ["a", "b", "c"]
y = x
y[1] = "z"
print(y)
print(x)

x = ["a", "b", "c"]
y = list(x)
y[1] = "z"
print(y)
print(x)
help(complex)
help(sorted)
# Create lists first and second
first = [11.25, 18.0, 20.0]
second = [10.75, 9.50]

# Paste together first and second: full
full = first + second

# Sort full in descending order: full_sorted
full_sorted = sorted(full, reverse = True)

# Print out full_sorted
print(full_sorted)
# string to experiment with: room
room = "poolhouse"

# Use upper() on room: room_up
room_up = room.upper()

# Print out room and room_up
print(room) ; print(room_up)

# Print out the number of o's in room
print(room.count("o"))
# Create list areas
areas = [11.25, 18.0, 20.0, 10.75, 9.50]

# Print out the index of the element 20.0
print(areas.index(20.0))

# Print out how often 14.5 appears in areas
print(areas.count(14.5))

# Use append twice to add poolhouse and garage size
areas.append(24.5)
areas.append(15.45)

# Print out areas
print(areas)

# Reverse the orders of the elements in areas
areas.reverse()

# Print out areas
print(areas)

# Definition of radius
r = 0.43

# Import the math package
import math

# Calculate C
C = 0
C = 2 * math.pi * r
# Calculate A
A = 0
A = math.pi * r * r

# Build printout
print("Circumference: " + str(C))
print("Area: " + str(A))
print("circumference: ", C)
height = [180, 215, 210, 210, 188, 176, 209, 200]
weight = [800, 900, 930, 1300, 1200, 1600, 1802, 1500]

# Import the numpy package as np
import numpy as np

# Create array from height with correct units: np_height_m
np_height_m = np.array(height) * 0.0254

# Create array from weight with correct units: np_weight_kg
np_weight_kg = np.array(weight) * 0.453592

# Calculate the BMI: bmi
bmi = np_weight_kg/(np_height_m * np_height_m)

# Print out bmi
print(bmi)

# Create the light array
light = bmi < 21

# Print out light
print(light)

# Print out BMIs of all baseball players whose BMI is below 21
print(bmi[light])
# Create baseball, a list of lists
baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]

# Import numpy
import numpy as np

# Create a 2D numpy array from baseball: np_baseball
np_baseball = np.array(baseball)

# Print out the 3rd row of np_baseball
print(np_baseball[2,:])

# Create np_height from np_baseball
np_height = np_baseball[:,0]

# Print mean height (first column)
avg = np.mean(np_baseball[:,0])
print("Average: " + str(avg))

# Print median height. Replace 'None'
med = np.median(np_baseball[:,0])
print("Median: " + str(med))

# Print out the standard deviation on height. Replace 'None'
stddev = np.std(np_baseball[:,0])
print("Standard Deviation: " + str(stddev))

# Print out correlation between first and second column. Replace 'None'
corr = np.corrcoef(np_baseball[:,0], np_baseball[:,1])
print("Correlation: " + str(corr))