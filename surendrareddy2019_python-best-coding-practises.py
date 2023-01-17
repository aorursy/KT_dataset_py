# List Variable
example = ['use','enumerate','instead','of','iteration']

# Ideal Way
for i in range(len(example)):
    print(f"# {i + 1}: {example[i]}")
          
# Pythonic way - enumerate
for i, value in enumerate(example, 1):
    print(f"# {i}: {value}")
# Lists 
Employees = ['Employee1','Employee2','Employee3','Employee4']
Age = [30,25,35,40]

# Ideal Way
for i in range(len(Employees)):
    employee = Employees[i]
    age = Age[i]
    print(f"Employee name is {employee} and age is {age}")
    
# Pythonic way - zip
for employee, age in zip(Employees, Age):
    print(f"Employee name is {employee} and age is {age}")
# Lists 
Employees = ['Employee1','Employee2','Employee3','Employee4']

# Ideal way
for i in range(1,len(Employees) + 1):
    print(f"Approach 1 - Employee came to office after covid 19 is {Employees[-i]}")
for employee in Employees[::-1]:
    print(f"Approach 2 - Employee came to office after covid 19 is {employee}")
    
# Pythonic way - reversed()
for employee in reversed(Employees):
    print(f"Using revered -  Employee came to office after covid 19 is {employee}")
# List
numbers = [1,2,3,4,5,6,7,8,9,10]

#Ideal way
for number in numbers:
    if number % 2:
        print(f"Odd Number : {number}")

# Pythonic way - filter()
for number in filter(lambda x: x %2, numbers):
    print(f"Odd Number : {number}")             
from itertools import chain

#Lists
oddValues = [1,3,5,7,9]
evenValues = [2,4,6,8,10]

# Ideal way
values = oddValues + evenValues
for value in values:
    print(f"value is : {value}")

# Pythonic way - chain()
for value in chain(oddValues, evenValues):
    print(f"value is : {value}")
# Dict
Employees = {"Employee1": 30, "Employee2": 35, "Employee3": 40, "Employee4": 45}

#Ideal way
for key in Employees:
    print(f"Employee Name is : {key}")
for key in Employees.keys():
    print(f"Employee Name is : {key}")
for value in Employees.values():
    print(f"Age is : {value}")
for value in Employees:
    print(f"Age is : {Employees[value]}")
    
#Pythonic way
for key, value in Employees.items():
    print(f"Employee came to office after covid 19 is {key} and age is {value}")
    
### list
numbers = [1,2,3,4,5,6,7,8,9,10]

#Ideal way
squaredNumbers = list()
for square in numbers:
    squaredNumbers.append(square * square)
print(squaredNumbers)

#Using list comprehension
squaredNumbers = [x * x for x in numbers]
print(squaredNumbers)

#Ideal way
squaredNumbers = dict()
for square in numbers:
    squaredNumbers[square] = square * square
    
#Using list comprehension
squaredNumbers = {x: x*x for x in numbers}
print(squaredNumbers)

#Ideal way
squaredNumbers = set()
for square in numbers:
    squaredNumbers.add(square)
print(squaredNumbers)

#Using list comprehension
squaredNumbers = [x*x for x in numbers]
print(squaredNumbers)    
# For Loop
for n in range(2, 10):
    for x in range(2, n):
        if n % x == 0:
            print( n, 'equals', x, '*', n/x)
            break
    else:
        # loop fell through without finding a factor
        print(n, 'is a prime number')

# While Loop
count = 2
while (count < 1):     
    count = count+1
    print(count) 
    break
else: 
    print("No Break")
    #Traditional
    value = True
    if value:
        v = 1
    else:
        v = 0
    print(v)

    #Using ternary
    value = True
    v = 1 if value else 0
    print(v)
    # Tip1: Accept Multiple Inputs

    # Traditional Approach

    x = input("Enter Any Number: ")
    print(x)

    y = input("Enter Any Number: ") 
    print(y)

    z = input("Enter Any Number: ") 
    print(z)

    t = input("Enter Any Number: ") 
    print(t)

    p = input("Enter Any Number: ") 
    print(p)

    # Pythonic way
    x,y,z,t,p = input("Enter Any Number: ").split(' ')
    print(x,y,z,t,p)

    # Tip2: Multi Condition Check

    salary = 40000
    age = 25
    weight = 70

    # Traditional Approach

    if salary > 20000 and age > 20 and weight > 65:         
    	print ("All conditions satisfied")

    if salary > 20000 or age > 20 or weight > 65:
    	print ("Any one condition is satisfied")

    # Pythonic way using list

    check = [
    	salary > 20000,
    	age > 20,
    	weight > 65
    ]

    if all(check):
    	print("Pythonic way of checking conditions")

    if any(check):
    	print("Pythonic way of checking any one condition")

    # Tip 3: swapping in python

    # Traditional approach

    x = "tip1"
    y = 'tip2'

    temp = x
    x = y
    y = temp

    print(x,y)

    # Pythonic way
    x = 'tip3'
    y = 'tip4'
    x,y = y,x
    print(x,y)

    # Tip 4: Removing duplicates

    # Traditional approach - with out list comprehension
    numbers = [1,2,1,3,4,2,1,2,5,67,2,3,56,78,34,12,3,4,5,6,7,8]
    result = []
    for num in numbers:
    	if num not in result:
    		result.append(num)
    print("final list is :" + str(result))

    # with list comprehension
    resultComp = []
    [resultComp.append(num) for num in numbers if num not in resultComp]
    print("final list using comprehension:" + str(resultComp))

    # Pythonic way

    # Using set
    resultSet = list(set(numbers))
    repeatedNumbers = max(set(numbers), key=numbers.count)
    print("Using set final list is :" + str(resultSet))
    print("Most repeated is :", repeatedNumbers)

    # Using Dictionary
    resultDict = list(dict.fromkeys(numbers))
    repeatedNumbers = max(dict.fromkeys(numbers), key=numbers.count)
    print("Most repeated is :", repeatedNumbers)
    print("Using dictionary final list is :" + str(resultDict))

    # Tip 5: Call by Reference

    # Traditional way of implementing the sum function
    def finalString(x,y):
    	return x + y
    print(finalString('Python is very simple',' to learn'))

    # Using Pythonic way

    def finalString1(*x):
    	result = ''
    	for s in x:
    		result += s
    	return result

    print(finalString1('Python is very simple', ' to learn', ' and can be used \
    in', ' Test Automation, Machine learning, Data Science, Web', \
    'desktop apps development.'))

    # Tip 6: Reverse String

    # Traditional approach
    s = 'python is fun to learn'
    print(s[::-1])

    # Pythonic way
    s = 'python is fun to learn'[::-1]
    print(s)

    # Tip 7: Palidrome

    checkString = input("Enter the string value :")
    result = checkString.find(checkString[::-1])==0
    if result:
    	print("String is Palindrome " + str(result))
    else:
    	print("Not Palindrome")    