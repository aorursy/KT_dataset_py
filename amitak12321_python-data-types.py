#print statment is used to print any value in python
print(True,False)
#type is used to print the data type of the arguments
type(True)
type(False)
#declaring a string variable my_str with value AMit Kumar
my_str='Amit Kumar'
#The istitle() returns True if the string is a titlecased string otherwise it returns False.
#titlecased:- String which has the first character in each word Uppercase and remaining all characters Lowercase alphabets.
my_str.istitle()
print(my_str.isalnum()) #check if all char are numbers
print(my_str.isalpha()) #check if all char in the string are alphabetic
print(my_str.isdigit()) #test if string contains digits
print(my_str.istitle()) #test if string contains title words
print(my_str.isupper()) #test if string contains upper case
print(my_str.islower()) #test if string contains lower case
print(my_str.isspace()) #test if string contains spaces
print(my_str.endswith('r')) #test if string endswith a d
print(my_str.startswith('A')) #test if string startswith H
True and True
True and False
True or False
True or True
str_example='Hello World'
my_str='Amit'
my_str.isalpha() or str_example.isnum()
type([])
#creating a empty list
lst_example=[]
type(lst_example)
lst=list()
type(lst)
lst=['Mathematics', 'chemistry', 100, 200, 300, 204]
len(lst)
type(lst)
#.append is used to add elements in the list
#adding one element
lst.append("Amit")
#adding multiple elements to a list
lst.append(["John","Bala"])
lst
##Indexing in List
lst[6]
lst[1:6]
## insert in a specific order

lst.insert(2,"Kumar")
lst
lst.append(["Hello","World"])
lst
lst=[1,2,3]
lst.append([4,5])
lst
lst=[1,2,3,4,5,6]
lst.extend([8,9])
lst
lst=[1,2,3,4,5]
sum(lst)
lst*5
#drop the last value from the list
lst.pop()
lst
lst.pop(0)
lst
lst=[1,1,2,3,4,5]
lst.count(1)
#length:Calculates total length of List
len(lst)
# index(): Returns the index of first occurrence. Start and End index are not necessary parameters
#syntex index(element.start,end)
lst.index(2,1,5) 
##Min and Max
min(lst)
max(lst)
## Defining an empy set

set_var= set()
print(set_var)
print(type(set_var))
#creating a set
set_var={1,2,3,4,3}
set_var
set_var={"Avengers","IronMan",'Hitman','Antmman'}
print(set_var)
type(set_var)
## Inbuilt function in sets

set_var.add("Hulk")

print(set_var)
set1={"Avengers","IronMan",'Hitman'}
set2={"Avengers","IronMan",'Hitman','Hulk2'}
set2.intersection_update(set1)
set2
##Difference 
set2.difference(set1)
set2
## Difference update

set2.difference_update(set1)
print(set2)
dic={}
type(dic)
type(dict())
set_ex={1,2,3,4,5}
type(set_ex)
## Let create a dictionary

my_dict={"Car1": "Audi", "Car2":"BMW","Car3":"Mercidies Benz"}
type(my_dict)
##Access the item values based on keys

my_dict['Car1']
# We can even loop throught the dictionaries keys

for x in my_dict:
    print(x)
# We can even loop throught the dictionaries values

for x in my_dict.values():
    print(x)
# We can also check both keys and values
for x in my_dict.items():
    print(x)
## Adding items in Dictionaries

my_dict['car4']='Audi 2.0'
my_dict
my_dict['Car1']='MAruti'
my_dict
car1_model={'Mercedes':1960}
car2_model={'Audi':1970}
car3_model={'Ambassador':1980}

car_type={'car1':car1_model,'car2':car2_model,'car3':car3_model}
print(car_type)
## Accessing the items in the dictionary

print(car_type['car1'])
print(car_type['car1']['Mercedes'])
## create an empty Tuples

my_tuple=tuple()
type(my_tuple)
my_tuple=()
type(my_tuple)
my_tuple=("Amit","Krish","Ankur","John")
my_tuple=('Hello','World',"Amit")
print(type(my_tuple))
print(my_tuple)
type(my_tuple)
## Inbuilt function
my_tuple.count('Amit')
my_tuple.index('Amit')
