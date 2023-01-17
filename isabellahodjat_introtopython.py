print("hello world") 

#lines in a code cell with a hashtag in front will not run so can be used to comment on what the code does
#let's do some variable assignments

a = 3

b = "hello world"



#now let's try printing out some variables

print(a)

print(b)

print(52)

print("hi")
a=2

a=5

print(a)
import numpy as np #don't worry about this line for now, it will be explained in the packages section below. 



data = np.array([3.1, 3.2, 3.0, 3.1])

# read as: "data is assigned an array with 3.1, 3.2, 3.0, 3.1"



print(data)

x = np.array([1.1, 2.0, 3.3, 3.9, 4.8, 6.0]) #put data between the square brackets [], with each entry separated by a comma

y = np.array([4.3, 7.9, 12.5, 15.1, 19.2, 23.1]) #finish this line



print(x)

print(y)
#don't worry about what this code means, just check that this method works

#we will provide all code for loading CSVs in the future

csv_data = []

import csv

file = open("IntrotoPython.csv",encoding='utf-8-sig')

reader = csv.reader(file)

for row in reader:

    csv_data.append(row)

file.close()

csv_data = np.array(csv_data)



x = csv_data[:,0]

y = csv_data[:,1]



print(x)

print(y)
print("Isabella Hodjatzadeh")
#this is the function definition, which will usually be given to you

def make_waffles(flour, water):

    print("you gave me:", flour, "in the variable flour")

    print("you gave me:", water, "in the variable water")

    print("preparing your waffle")

    print("giving you back:")

    return "a rather simple waffle"
#this is the function call

make_waffles("wheat flour", "water from the tap")
#a function call that prints the output

print(make_waffles("wheat flour", "water from the tap"))
make_waffles("water from the tap","wheat flour")
make_waffles(water="water from the tap",flour="wheat flour")
make_waffles(orange="water from the tap",flour="wheat flour")
def make_pancakes(flour, water, baking_soda="the one from the pantry", add_ins = "bananas"):

    print("you gave me:", flour, "in the variable flour")

    print("you gave me:", water, "in the variable water")

    print("you gave me:", baking_soda, "in the variable baking_soda")

    print("you gave me:", add_ins, "in the variable add_ins")

    print("doing pancake things")

    print("giving you back:")

    return "a pancake"

make_pancakes("wheat flour", "water from the tap")
make_pancakes("wheat flour", "water from the tap", add_ins = "chocolate chips")

make_pancakes("wheat flour", baking_soda="the new one from the store")
make_pancakes("filtered water", "chocolate chips and bananas", baking_soda="the one from the pantry")
import numpy

import matplotlib.pyplot
numpy.sqrt(4)
import numpy as np

import matplotlib.pyplot as plt
np.sqrt(4)
plt.plot(x,y, 'ro')

plt.show()
my_data = np.array([2.1, 2.3, 2.6, 2.4]) #create a numpy array here



def average(input_data):

    #this function takes a data set and returns the average of the data

    return np.sum(input_data)/len(input_data)



average(my_data)
more_data = np.array([3.3,3.4,3.2])



def difference_between_averages(data1, data2):

    #this function finds the averages of the two data sets and returns the absolute value 

    #of the difference between the two averages

    avg1 = np.sum(data1)/len(data1)

    avg2 = np.sum(data2)/len(data2)

    return np.abs(avg1-avg2)



difference_between_averages(my_data, more_data)
# define your own data set:

my_data2= np.array([3.2, 3.5, 3.8, 4.2])

# write a function call:

def difference_between_averages(my_data, my_data2):

    avg1 = np.sum(my_data)/len(my_data)

    avg2 = np.sum(my_data2)/len(my_data2)

    return np.abs(avg1-avg2)



difference_between_averages(my_data, my_data2)
