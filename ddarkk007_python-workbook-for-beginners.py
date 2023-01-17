#Numeric Variable



variable1 = 10

variable2 = 15

variable3 = 20

variable4 = 40

variable5 = 55



#Double Numeric Variable



variable6 = 30.5

variable7 = 40.4

variable8 = 10.3

variable9 = 19.7
print(variable1 + variable2)
print((variable3+variable5)/(variable7+variable9))
#String Variable



s = 'It\'s a cat'

variable_type = type(s)

print(variable_type)
strvar1 = 'Hello! '

strvar2 = 'My name is '

strvar3 = 'Enes'



print(strvar1+strvar2+strvar3)
strvar4 = 'string variable'

print(len(strvar4))
strvar5 = 'string variable character'

print(strvar5[0])
#String to integer and float!



strvar10 = '400'

int(strvar10) #change integer

print(float(strvar10)) #and change float
#Returns the absolute value of a number



strvar11 = -500.8

abs(strvar11)
#Returns True if all items in an iterable object are true

#ERROR: TypeError: 'int' object is not iterable



strvar12 = 'string variable 12'

all(strvar12)
#Returns True if any item in an iterable object is true



strvar13 = 'string variable 13'

any(strvar13)
#Returns a readable version of an object. Replaces none-ascii characters with escape character



strvar14 = 'string varıable 14'

ascii(strvar14)
#Returns the boolean value of the specified object

#1:True and 0:False



strvar16 = 1

bool(strvar16)
#Returns a floating point number



numericvar1 = 1000

float(numericvar1)
#Returns an integer number



numericvar2 = 250.4

int(numericvar2)
#Returns the length of an object



strvar17 = 'it is string variable'

len(strvar17)
#Returns the smallest item in an iterable



strvar18 = '465329'

min(strvar18)
#Returns the highest item in an iterable



strvar19 = '465329'

max(strvar18)
#Returns a new string



numericvar4 = 400

str(numericvar4)
#Returns the type of an object



numericvar5 = 700

type(numericvar5)
#user defined functions



variable20 = 10

variable21 = 100



output = ((variable20+variable21)*100)/400 + variable20

print(output)
def calculator_function(variable20,variable21):

  """

  parameter: variable20, variable21

  return: Calculation of the given values

  """

  output_function = ((variable20+variable21)*100)/400 + variable20

  return output_function

#Call User Defined Function (calculator_function)

calculator_function(40,50)
#Default Function:

#Calculating the circumference of the circle



def circle_calculating(r,pi = 3.14):

  """

  input: r, pi

  return: calculating the circumference of the circle

  """

  output_circle = 2*pi*r

  return output_circle
#Call function

circle_calculating(2)
#Flexible Function:

"""def calculate(weight, height, *args):

  print(len(args))

  output_w = weight + height

  return output_w"""



def calculate(weight, height, *args):

  print(args)

  output_w = (weight + height)*args[1]

  return output_w
#Call function

#*args = 22,44

calculate(60,120,22,44)
#int variable; age

#string variable; name

#fuction; print(type(), len(), float())

#*args: surname

#default parameter: shoes number



name = 'Enes'

surname = 'Akkaya'

age = 22



def function_quiz(name,surname,age,shoes_number = 44,*args):

  print(args)

  print(type(name))

  print(float(age))

  print(len(surname))

  print(name, 'is', age ,'years old.')



  output_quiz = (age + shoes_number)*args[0]

  return output_quiz



#call function



function_quiz(name,surname,age,44,24,42,45)
#Profit and Loss Calculation



def function_quiz1(cost,sale_price,tax = 0.18):

  """

  input: cost, sale_price, tax

  return: output_quiz1

  """

  print('Cost Price:', cost)

  print('Sale Price:', sale_price)

  print('The Tax(%)', tax)



  output_quiz1 = sale_price - ((sale_price*tax) + cost)

  print('Net Remaining($):', output_quiz1)





  if output_quiz1 > cost:

    print('It made a profit.')

  elif output_quiz1 == cost:

    print('No profit or loss.')

  else:

    print('The business has damaged.')



#call function



function_quiz1(100,400)

print('#####################')

function_quiz1(100,200)
#Enables us to write functions faster than other functions



output2 = lambda x: x*x

output2(3)
#There are four collection data types in the Python programming language:



#List is a collection which is ordered and changeable. Allows duplicate members.

#Tuple is a collection which is ordered and unchangeable. Allows duplicate members.

#Set is a collection which is unordered and unindexed. No duplicate members.

#Dictionary is a collection which is unordered, changeable and indexed. No duplicate members.



#List is a collection which is ordered and changeable. Allows duplicate members.

#Create a List



new_list = [1,2,3,4,5,'apple','cherry',6,7,8,9]

print(type(new_list))



#Print list elements



value0 = new_list[0] #Call element 0

print(value0)



value6 = new_list[6] #Call element 6

print(value6)



last_value = new_list[-1] #Call last element

print(last_value)



value03 = new_list[0:3] #Invoke values between 0 and 3 elements

print(value03)



value_reverse = new_list[::-1] #Reverse list elements

print(value_reverse)
#List Function (Built-in Functions):



dir(list)
#List Function (Built-in Functions) Example:



new_list3 = [1,2,3,4,5,6,7,8,9]



new_list3.append(10) #Create a new element

print(new_list3)



copy_list = new_list3.copy() #List a copy

copy_list.remove(4) #Remove 4 element in copy_list

print(copy_list)



copy_list.reverse() #Reverse a list

print(copy_list)



x = new_list3.index(2) #Index is the value input element!

print(x)
#A dictionary is a collection which is unordered, changeable and indexed. 

# In Python dictionaries are written with curly brackets, and they have keys and values.



#Create a dictionary

new_dictionary = {'model_number': 23842,

                  'model_name': 'XD-MODEL',

                  'sale_price': 300.50,

                  'cost': 100}



print(new_dictionary['model_number'])
#Dictionary add new element



new_dictionary['color'] = 'red'

new_dictionary
#Dictionary remove a element



new_dictionary.pop('color')

new_dictionary
#The popitem() method removes the last inserted item (in versions before 3.7, a random item is removed instead):



new_dictionary1 = {'color': 'blue',

                   'number': 222,

                   'lucky number': 20.5,

                   'name': 'serhat'}

x = new_dictionary1        

print(x)
#The del keyword removes the item with the specified key name:



new_dictionary2 = {'color': 'blue',

                   'number': 222,

                   'lucky number': 20.5,

                   'name': 'serhat'}



del new_dictionary2['name']

print(new_dictionary2)
#The del keyword can also delete the dictionary completely:



del new_dictionary2
#The clear() method empties the dictionary:



new_dictionary3 = {'color': 'blue',

                   'number': 222,

                   'lucky number': 20.5,

                   'name': 'serhat'}



new_dictionary3.clear()

print(new_dictionary3)
#Make a copy of a dictionary with the copy() method:



new_dictionary4 = {'color': 'blue',

                   'number': 222,

                   'lucky number': 20.5,

                   'name': 'serhat'}



copy_dictionary4 = new_dictionary4.copy()

print(copy_dictionary4)
#Dictionary built-in function; dict()



new_dictionary5 = {'color': 'blue',

                   'number': 222,

                   'lucky number': 20.5,

                   'name': 'serhat'}



copy_dictionary5 = dict(new_dictionary5)

print(copy_dictionary5)
#Create a dictionary that contain three dictionaries:



sport_members = {

    "member1" : {

        "name" : "Lokman",

        "age" : 22

    },

    "member2" : {

        "name" : "Serhat",

        "age" : 25

    },

    "member3" : {

        "name" : "Emir",

        "age" : 30

    }

}



print(sport_members)
#It is also possible to use the dict() constructor to make a new dictionary:



new_dict = dict(brand='xforge',name='john',age=20)

print(new_dict)
#A tuple is a collection which is ordered and unchangeable. In Python tuples are written with round brackets.



new_tuple = ("banana","apple","orange","cherry","strawberry","pineapple")
dir(tuple)
#Call the first element in the group

new_tuple[1]
#Call the last element of the group



new_tuple[-1]
#Calling 0 to 3 elements in the group



new_tuple[0:3]
#return number of occurrences of value



new_tuple.count('cherry')
#A set is a collection which is unordered and unindexed. In Python sets are written with curly brackets.



new_set = {"john","alex","dennis"}

new_set
#Check if "alex" is present in the set:



print("alex" in new_set)
#Add an item to a set, using the add() method:



new_set.add("serhat")

new_set
#Add multiple items to a set, using the update() method:



new_set.update(["omer","ali","derya","lokman"])

new_set
#Get the number of items in a set:



print(len(new_set))
#To remove an item in a set, use the remove(), or the discard() method.



new_set.remove('lokman')

new_set.discard('omer')

new_set
#The union() method returns a new set with all items from both sets:



new_set1 = {'lokman','serhat','omer','emir'}

new_set2 = {'yasin','enes','ibrahim'}



new_set3 = new_set1.union(new_set2)

new_set3
#Returns whether two sets have a intersection or not



new_set4 = {'lokman','serhat','omer'}

new_set5 = {'enes','ali','ibrahim'}



y = new_set4.isdisjoint(new_set5)

print(y)
#Returns a set, that is the intersection of two other sets



new_set6 = {'lokman','serhat','omer'}

new_set7 = {'serhat','omer','ibrahim'}



z = new_set6.intersection(new_set7)

print(z)
#What is the conditionals



print(1 == 1)

print(1 == 2)

print(0 != 1)

print(4 > 5)

print(10 < 11)
#If-else statements



var100 = 25

var101 = 24.9



if (var100 > var101):

  print("Correct!")

elif(var100 == var101):

  print("Equal!")

else:

  print("Wrong!")
#Transforming centuries by years

#Output int



def years_change(year):

  """

  input: year

  output: Transforming centuries by years

  """

  str_year = str(year)



  if (len(str_year) < 3):

    return 1

  elif (len(str_year) == 3):

    if (str_year[1:3] == "00"):

      return int(str_year[0])

    else:

      return int(str_year[0]) + 1

  else:

    if (str_year[2:4] == "00"):

      return int(str_year[0:2])

    else:

      return int(str_year[0:2]) + 1



print(years_change(100))

print(years_change(88))

print(years_change(1000))

print(years_change(245))

print(years_change(1004))
for each in range(1,12): #1: inclusive 12: exclusive

  print(each)
for each in 'john and alice':

  print(each)
for each in 'john and alice'.split():

  print(each)
for_list = [1,2,3,4,5,6,7,8,9,10]



sum_for_list = sum(for_list)



count = 0

for each in for_list:

  count = count + each

  print(count) 
i = 0

while (i < 5):

  print(i)

  i = i + 1
while_list = [1,2,3,4,5,6,7]



s = len(while_list)

each = 0

count_w = 0



while (each < s):

  count_w = count_w + while_list[each]

  each = each + 1



print(count_w)
#Find the lowest value in the list



quiz_list1 = [1,3,4,53,7556,4534,-304,-103,-645,0,43]



min1 = 10000



for each in quiz_list1:

  if (each < min1):

    min1 = each

  else:

    continue



print(min1)
#Employee list



class Employee:



  def __init__(self,name,surname,age,departmant):

    self.name = name

    self.surname = surname

    self.age = age

    self.departmant = departmant

  

  def giveNameSurname(self):

    return self.name + " " + self.surname



employee1 = Employee("John","David",22,"Marketing")

print(employee1)

print(employee1.giveNameSurname())
#Stundents list



class Students:



  score_coefficient = 0.15

  counter = 0 #We will use it to learn the number of students.

  def __init__(self,stname,stsurname,stage,gender,exam_score,performance_score):

    self.stname = stname

    self.stsurname = stsurname

    self.stage = stage

    self.gender = gender

    self.exam_score = int(exam_score)

    self.performance_score = performance_score



    Students.counter = Students.counter + 1 #We will use it to learn the number of students.

  

  def studentNameSurname(self):

    return self.stname + " " + self.stsurname



  def giveGender(self):

    return self.stname + " " + self.stsurname + " is " + self.gender + "!"

  

  def updatePerformanceScore(self):

    self.performance_score = (self.exam_score * self.score_coefficient) + self.performance_score



student1 = Students("Jack","Dennis",19,"Male",70,60)

print(student1)

print(student1.giveGender())
#Class Variable

#We have created our student class above and we want to update these students' performance grades.



student2 = Students("John","Oracle",22,"Male",70,65)

print("Student's Name:", student2.stname)

print("Student's Surname:", student2.stsurname)

print("How old is he?:", student2.stage)

print("Student's Gender:", student2.gender)

print("Student's Exam Score:", student2.exam_score)

print("Student's Performance Score:", student2.performance_score)

student2.updatePerformanceScore() #updated performance

print("Updated Student's Performance Score:", student2.performance_score)
#Create a new student

student3 = Students("Kate","Lose",20,"Female",70,55)

student4 = Students("Angelica","Rose",27,"Female",90,60)
#How many students do we have?

Students.counter
#Create a list

students_list = [student1,student2,student3,student4]

students_list
#Top rated student?



max_examscore = -1



for each in students_list:

  if (each.exam_score > max_examscore):

    max_examscore = each.exam_score

    index = each

print(max_examscore)



#The highest grade is determined, but who got it?

print(index.studentNameSurname())



#import

import numpy as np
#array

array = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

array
#shape and reshape



print("array's shape:", array.shape)

a = array.reshape(3,5)

print(a)



#dimension

print("dimension array:", array.ndim)

print("dimension a:", a.ndim)



#value is data type

print("data type:", a.dtype.name)



#size

print("size:", a.size)



#type function

print(type(a))
#New array 2D



array1 = np.array([[2,3,4,5,6],[7,8,9,10,11]])

array1



print(array1.shape)
#np.zeros



zeros = np.zeros((4,5))



#zeros update

zeros[0,0] = 5

print(zeros)
#np.ones



np.ones((3,4))
#np.empty



# p = np.empty((3,4))

# print(p)
#np.arange



np.arange(10,50,5)
#np.linspace



np.linspace(10,50,20)
#np.random



c = np.random.random((5,5))

print(c.sum())

print(c.max())

print(c.min())

print(c.sum(axis=0))

print(c.sum(axis=1))
#np.sqrt and np.square



print(np.sqrt(c))

print("#######################")

print(np.square(c))
#np.add



print(np.add(c,c))
#Basic operations



array5 = np.array([1,2,3,4,5,6,7,8,9])

array6 = np.array([10,11,12,13,14,15,16,17,18])



print(array5 + array6)

print(array5 - array6)

print(array5**2)

print(np.sin(array5))

print(array5 < 6)

print(array5 * array6)
#Indexing and Slicing



array_new = np.array([4,5,6,7,8,9])

print(array_new)



#Call Index

print(array_new[2])

print(array_new[0:3])



#Reverse

reverse_array_new = array_new[::-1]

print(reverse_array_new)



#New array 2d

array_new_2d = np.array([[2,3,4,5],[6,7,8,9]])

print(array_new_2d)



#Call Index 2d

print(array_new_2d[1,1])

print(array_new_2d[:,1])

print(array_new_2d[1,1:4])

print(array_new_2d[-1,:])
#Shape Manipulation



array_new_3d = np.array([[2,3,4,5],[6,7,8,9],[10,11,12,13]])

print(array_new_3d)
#ravel command

f = array_new_3d.ravel()

f
#Return

g = f.reshape(4,3)

g
#reshape vs resize



array99 = np.array([[2,3,4,5],[6,7,8,9]])

print(array99.reshape(4,2))

print(array99) #not change!

print("#################")

#resize

array100 = np.array([[1,2,3,4],[5,6,7,8]])

array100.resize((4,2))

print(array100) #change!
#Stacking Arrays



array101 = np.array([[1,2,3,4],[5,6,7,8]])

array102 = np.array([[-1,-2,-3,-4],[-5,-6,-7,-8]])



print(array101)

print('----------')

print(array102)

print('---------- Vertical')



#vertical combining

array103 = np.vstack((array101,array102))

print(array103)

print('---------- Horizontal')



#horizontal combining

array104 = np.hstack((array101,array102))

print(array104)
#Convert and Copy Array



list_new94 = [1,2,3,4,5]

array_listnew = np.array([5,6,7,8])



#Convert to array

array_listnew94 = np.array(list_new94)

print(array_listnew94)



#Convert to list

list_new95 = list(array_listnew)

print(list_new95)
#Create 5 numeric and string variables



#numeric variable

numeric_variable1 = 10

numeric_variable2 = 20.5

numeric_variable3 = 40

numeric_variable4 = 50

numeric_variable5 = -100



#string

string_variable10 = "apple"

string_variable20 = "200"

string_variable30 = "banana"

string_variable40 = "country"

string_variable50 = "500"



#operations and printing



print(numeric_variable1)



printed_value = ((numeric_variable1 * numeric_variable3) + (numeric_variable5 + numeric_variable2) + 1000)

print(printed_value)



print(string_variable10, string_variable30)

print(string_variable20 + string_variable50)
#built-in function example



#Change integer to float

value_int = 100

print(float(value_int))



#Change float to integer

value_float = 20.5

print(int(value_float))



#type function

value_string = "it is a banana"

print(type(value_string))



#len function

print(len(value_string))



#abs function

value_abs = -100

print(abs(value_abs))



#string upper function

print(value_string.upper())



#string title function

print(value_string.title())
#user defined function

#Automatically multiplying the two values and dividing the resulting value by 100



def auto_calculate(a_variable,b_variable):

  output_variable = (a_variable * b_variable) / 100

  return output_variable



auto_calculate(100,200)
#default and flexible function

#Calculate the division of the two values automatically multiplied by 100 and then make the sum with the standard given value. In addition, if the user has entered an extra value, the extra value entered must be subtracted from the number.



def auto_calculate2(c_variable,d_variable,f_variable = 40):

  output_variable2 = ((c_variable * d_variable) / 100) + f_variable

  return output_variable2



auto_calculate2(200,200)



#flexible function

"""

Calculate the division of the two values automatically multiplied by 100 and then make the sum with the standard given value. 

In addition, if the user has entered an extra value, the extra value entered must be subtracted from the number.

"""



def auto_calculate3(x_variable,y_variable,z_variable = 50, *args):

  print(args)

  output_variable3 = (((x_variable * y_variable) / 100) + z_variable) - args[0]

  return output_variable3



auto_calculate3(100,200,50,40)
#Data type; list



datatype_list = [1,2,3,4,5,6,7,8,9,10]



#list indexing

print(datatype_list[0])

print(datatype_list[1:3])

print(datatype_list[-1])

print(datatype_list[-4:-1])

print(datatype_list[::-1])



#list function

print("-----------")



datatype_list.append(11)

print("Append Function:", datatype_list)



datatype_list.remove(7)

print("Remove Function:", datatype_list)



datatype_listcopy = datatype_list.copy()

print("Copy Function:", datatype_listcopy)



datatype_listcopy.clear()

print("Clear Function:", datatype_listcopy)



datatype_list.sort()

print("Sort Function:", datatype_list)



datatype_list.count(6)

print("Count Function:", datatype_list)



datatype_list.pop(0)

print(datatype_list)
#Data type; dictionary



datatype_dictionary = {"Name": "John",

                       "Surname": "David",

                       "Age" : 19,

                       "Gender": "Male",

                       "Job": "Doctor",

                       "Lucky Number": 20}



print("Dictionary:", datatype_dictionary)



#dictionary indexing!



print(datatype_dictionary["Name"])

print(datatype_dictionary["Age"])

datatype_dictionary["John's Father"] = "Mike"

print(datatype_dictionary)
#Data type; tuple



datatype_tuple = (1,2,3,4,5,6,7,8,9,10)



print("Tuple:", datatype_tuple)



#tuple indexing

print(datatype_tuple[0])

print(datatype_tuple[3])

print(datatype_tuple.count(2))
#Data type; sets



datatype_set = {1,2,3,4,5,6,7,8}

datatype_set2 = {9,10,11,7,13,14,15,16}



print("Set1:", datatype_set)

print("Set2:", datatype_set2)



#sets indexing

print(datatype_set.union(datatype_set2))

print(datatype_set.intersection(datatype_set2))
#Example 1



if (2 > 1):

  print('True')

else:

  print('False')
#Example 2



def examScore(student_grade1, student_grade2, student_grade3, student_grade4):

  

  print("Stundent Grades")

  print("1. Student Grade:", student_grade1)

  print("2. Student Grade:", student_grade2)

  print("3. Student Grade:", student_grade3)

  print("4. Student Grade:", student_grade4)

  print("-----------------------------------")



  student_gradelist = [student_grade1, student_grade2, student_grade3, student_grade4]

  average_grade = (student_grade1 + student_grade2 + student_grade3 + student_grade4) / len(student_gradelist)

  print("Average Grade:", average_grade)



  for each in student_gradelist:

    if (each > average_grade):

      print("Student grade levels are generally good!")

    elif (each == average_grade):

      print("Student grade levels are generally not bad!")

    else:

      print("Student grade levels are generally poor")



examScore(50,60,70,40)
#Workers list



class Workers:



  workers_counter = 0

  def __init__(self, worker_name, worker_surname, worker_age, worker_gender, worker_position, worker_salary):

    self.worker_name = worker_name

    self.worker_surname = worker_surname

    self.worker_age = worker_age

    self.worker_gender = worker_gender

    self.worker_position = worker_position

    self.worker_salary = worker_salary



    Workers.workers_counter = Workers.workers_counter + 1



  def workersNameSurname(self):

    return self.worker_name + " " + self.worker_surname

  

  def updateWorkerSalary(self):

    self.worker_salary = (self.worker_salary * 0.18) + self.worker_salary



worker1 = Workers("Alex", "Spain", 22, "Male", "Marketing", 700)

worker2 = Workers("Sam", "Candle", 25, "Male", "Partner", 2000)

worker3 = Workers("Steve", "John", 19, "Male", "Student", 400)



print("Q: How many workers are there?:")

print("A:", Workers.workers_counter)

print(" ")

print("Q: What is the name and surname of worker number 1?:")

print("A:", worker1.workersNameSurname())

print(" ")

print("Q: The salary of worker number 2 will be raised!:")

worker2.updateWorkerSalary()

print("A:", worker2.worker_salary)
#import numpy



import numpy as np
#Create array



numpy1 = np.array([1,2,3,4,5,6,7,8])

numpy2 = np.array([[1,2,3,4],[5,6,7,8]])

numpy3 = np.array([[[1,2,3,4],[5,6,7,8],[9,10,11,12]]])



print(numpy1)

print(" ")

print(numpy2)

print(" ")

print(numpy3)
#ndim function



print(numpy1.ndim)

print(numpy2.ndim)

print(numpy3.ndim)
#ravel



numpy4 = numpy2.ravel()

print(numpy4)
#shape, reshape, resize



print(numpy1.shape)

print(numpy2.shape)

print(numpy3.shape)



print("---------------")

print(numpy2.reshape(4,2))

print(" ")

print(numpy3.reshape(2,2,3))



print("---------------")

numpy2.resize(4,2)

print(numpy2)

print(" ")

numpy3.resize(2,2,3)

print(numpy3)
#arange



numpy5 = np.arange(5,20)

print(numpy5)
numpy6 = np.linspace(5,10,25)

print(numpy6)
numpy7 = np.zeros((5,5))

print(numpy7)
numpy8 = np.random.random((5,5))

print(numpy8)
#import pandas

# Fast and effective for using dataframes

# .csv, .xlsx, .txt format

# Mising data NaN solve and handle

# We can use data effectively.

# Easy slicing and indexing

# Time series data

# A quick library

import pandas as pd

pd_dict = {"NAME" : ["JOHN", "ALEX", "DAVID"],

           "AGE" : [18, 16, 14],

           "MALE" : ["MALE", "FEMALE", "MALE"]}



#create a dataframe

pd_dict_convert = pd.DataFrame(pd_dict)

pd_dict_convert
#head method



pd_dict_convert.head()
#tail method

pd_dict_convert.tail()
#pandas basic methods



#columns

print("Columns:")

print(pd_dict_convert.columns)

print(" ")



#info

print("Info:")

print(pd_dict_convert.info())

print(" ")



#dtypes

print("dtypes:")

print(pd_dict_convert.dtypes)

print(" ")



#describe = numeric columns

print("describe:")

print(pd_dict_convert.describe())
#indexing and slicing



print(pd_dict_convert["NAME"])

print(" ")

print(pd_dict_convert.NAME)

print(" ")



#new feature (columns)

pd_dict_convert["MONEY"] = [100,200,300]

print(pd_dict_convert)

print(" ")



print(pd_dict_convert.loc[:, "AGE"])

print(" ")



print(pd_dict_convert.loc[0:3, "NAME"])

print(" ")



print(pd_dict_convert.loc[0:3, "NAME":"MALE"])

print(" ")



print(pd_dict_convert.loc[0:3, ["NAME","MALE"]])

print(" ")



print(pd_dict_convert[::-1])

print(" ")



print(pd_dict_convert.loc[:,:"AGE"])

print(" ")



print(pd_dict_convert.iloc[:,2])
#filtering pandas data-frame



# MONEY > 200

filter_pd = pd_dict_convert.MONEY > 200

print(filter_pd)



print(" ")

filter_data = pd_dict_convert[filter_pd]

print(filter_data)

print(" ")



filter2_pd = pd_dict_convert.AGE < 16

print(filter2_pd)



print(" ")

print(pd_dict_convert[filter_pd & filter2_pd])

print(" ")



print(pd_dict_convert[pd_dict_convert.MONEY > 200])
#list comprehension



mean_pd = pd_dict_convert.MONEY.mean()

print(mean_pd)

print(" ")



pd_dict_convert["MONEY_LEVEL"] = ["HIGH" if mean_pd <= each else "LOW" for each in pd_dict_convert.MONEY]

print(pd_dict_convert)

print(" ")



pd_dict_convert.columns = [each.lower() for each in pd_dict_convert.columns]

print(pd_dict_convert)
#list comprehension



pd_dict_convert["money status"] = ["+","-","+"]



pd_dict_convert.columns = [each.split()[0] + "_" + each.split()[1] if (len(each.split()) > 1) else each for each in pd_dict_convert.columns]

pd_dict_convert
#concatenating data



pd_dict_convert.drop(["money_status"],axis=1, inplace= True)

print(pd_dict_convert)
#concatenating data



money = pd_dict_convert.money

gender = pd_dict_convert.male



money_gender = pd.concat([money,gender],axis=0, ignore_index= True)

print(money_gender)



money_gender2 = pd.concat([money,gender],axis=1)

print(money_gender2)
#transforming data



#apply

def multiply(money2):

  return money2*2



pd_dict_convert["money_x2"] = pd_dict_convert.money.apply(multiply)

pd_dict_convert

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))







df_new = pd.read_csv("/kaggle/input/iris-data/Iris.csv")
df_new.head()
df_new.tail()
df_new.describe()
df_new.info()
df_new.Species.unique()
setosa = df_new[df_new.Species == "Iris-setosa"]

versicolor = df_new[df_new.Species == "Iris-versicolor"]

virginica = df_new[df_new.Species == "Iris-virginica"]
print(setosa.describe())

print("---------------------")

print(versicolor.describe())

print("---------------------")

print(virginica.describe())
#Line plot



import matplotlib.pyplot as plt

df_new1 = df_new.drop(["Id"],axis=1)

df_new1
df_new1.plot()



plt.show()
#line plot 



plt.plot(setosa.Id, setosa.PetalLengthCm, color = "red", label ="Setosa PetalLengthCm")

plt.plot(versicolor.Id, versicolor.PetalLengthCm, color = "green", label ="Versicolor PetalLengthCm")

plt.plot(virginica.Id, virginica.PetalLengthCm, color = "orange", label ="Virginica PetalLengthCm")



plt.xlabel("Id")

plt.ylabel("PetalLengthCm")

plt.legend(loc="upper right")

plt.grid()

plt.show()
#scatter plot



plt.scatter(setosa.PetalLengthCm, setosa.PetalWidthCm, color='red', label='Setosa')

plt.scatter(versicolor.PetalLengthCm, versicolor.PetalWidthCm, color='green', label='Versicolor')

plt.scatter(virginica.PetalLengthCm, virginica.PetalWidthCm, color='orange', label='Virginica')



plt.xlabel("PetalLengthCm")

plt.ylabel("PetalWidthCm")

plt.legend(loc="upper right")

plt.show()
#histogram



plt.hist(setosa.SepalLengthCm)

plt.xlabel("SepalLengthCm Value")

plt.ylabel("Fre")

plt.title("Histogram")

plt.show()
#bar plot



new_numpy = np.array([1,2,3,4,5,6,7,8])



new_numpy1 = new_numpy*2 + 5



plt.bar(new_numpy, new_numpy1)

plt.show()
#subplots



df_new1.plot(subplots = True)



plt.show()