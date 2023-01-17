#defining dictionary with names and associated CGPA
dictionary={'riaz':4.0,'asad':3.8,'qammar':2.8}
#using key in for loop to  access names and associated CGPA with it
for key in dictionary:
#printing names and CGPA in  Dictionary
    print(key,dictionary[key])
#defining dictionary with names and associated CGPA
dictionary={'riaz':4.0,'asad':3.8,'qammar':2.8}
#filtering non-zero numbers
filtered = [v for _, v in dictionary.items() if v != 0]
#finding average CGPA
average=sum(filtered) / len(filtered)
#printing average CGPA of class
print(average)
#defining dictionary with names and associated CGPA
dictionary={'riaz':4.0,'asad':3.8,'qammar':2.8}
#using for loop to sort dictionary by key
for i in sorted(dictionary.keys()):
#printing sorted dictionary    
    print(i)
#importing matplotlib library
import matplotlib.pyplot as p
#defining dictionary
dictionary={'riaz':4.0,'asad':3.8,'qammar':2.8}
#using keys to get keys from dictionary
k=dictionary.keys()
#using values to get values from dictionary
v=dictionary.values()
#marking x-axis as names
p.xlabel('names')
#marking y-axis as CGPA
p.ylabel('CGPA')
#marking title of bar chart as CGPA chart
p.title('CGPA chart')
#ploting bar chart
p.bar(k,v)
#printing dictionary
dictionary

#importing numpy library
import numpy as n
#defining array of 64 elements
arr=n.arange(64)
#re-shaping the 2D array as 8,8
arr=arr.reshape(8,8)
#multiplying the array with last digit of my CMS-ID
arr=arr*2
#printing values at 4th row and 4th column
print(arr[3,3])

#importing numpy library
import numpy as n
#making array of 80 elements
arr=n.arange(80)
#re-shaping array as (5,4,4)
arr=arr.reshape(5,4,4)
#multipying array with last digit of my CMS-ID
arr=arr*2
#printing values and using sliciing to get first two values of 2nd row in 2nd matrix
print('after slicing',arr[1,1,0:2])




