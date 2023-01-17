#Chapter 1

#Comments are ignored by the computer and are displayed in a different color after the #.

#Comments help programmers understand code.



#data types:

#int: integers, whole numbers

#float: floating point values, decimals

#str: strings, characters(letters, symbols, numbers) strung together



#changing data types

#int(stuff) changes stuff into an int

#float(stuff) changes stuff into a float

#str(stuff) changes stuff into a string



#operators

# + :add for ints or floats

# + :concatenate(glue) for strings

# - :subtract

# * :multiply for ints or floats

# * :repeats or duplicates for int and a string

# ** :exponent

# / :regular decimal division

# // :integer division, returns integer with decimal cut off

# % :modular division, returns the remainder after division



# = :assignment, a=b should be read as "a gets b", a is assigned the value of b



print("Hello!") # prints Hello! to the screen

name=input("What is your name? ")   #prints What is your name? to the screen,  

                                    #waits for user to type stuff(Let's say Bob), 

                                    #then stores that stuff in a variable called name.



print("Your name is "+name+".") #If the user typed Bob, then this prints out Your name is Bob.

print() #prints a blank line



"""----------------You can also comment with 3 quotes, basically a non-executing string----------"""



#Chapter 2

#New data type

#bool: boolean value, can be True or False

True #reserved word, capitalization matters

False #reserved word, capitalization matters

#a>b,a<b, a>=b, a<=b, returns either true or false depending on the values of a and b



a=2 #assigns the integer value of 2 to the variable a

b=3 #assigns the integer value of 3 to the variable b

print("a is 2, b is 3\n") #prints a is 2, b is 3 to the screen.  \n inserts a new line

print("Is a greater than b? "+str(a>b))# prints Is a greater than b? False

                                       # I used str(a>b) because the glue, +, needs strings on

                                       # the left and right side

                                       





print("Is a the same as b? "+str(a==b))# prints Is a the same as b? False

                                       # Use == to see if two expressions are the same

                                       

print("Is a different from b? "+str(a!=b))# prints Is a different from b? True

                                          # Use != to see if two expressions are different.

                                          # Read a!=b as Is a not equal to b?



#Logical Operators

# a and b returns true only if both a and b are true

# a or b  returns false only if both a and b are false

print(a>0 and b>0) #True and True prints True

print(a>0 and b<0) #True and False prints False

print(a<0 and b>0) #False and True prints False

print(a<0 and b<0) #False and False prints False

print()

print(a>0 or b>0) #True or True prints True

print(a>0 or b<0) #True or False prints True

print(a<0 or b>0) #False or True prints True

print(a<0 or b<0) #False or False prints False

print()

print(not(a>0))# Since a is greater than zero evaluates to True,

               # then not(true) gives us False

print()

#Control Structures: if, if...else, if...elif...elif... what happens next is based on condition

someNumber=int(input("Type a number: "))

if someNumber>0:

    print("You typed a positive number!")

    if someNumber%2==0:

        print("And your number is divisible by two!")

    else:

        print("And your number is NOT divisible by two!")

elif someNumber<0:

    print("You typed a negative number!")

    if someNumber%2==0:

        print("And your number is divisible by two!")

    else:

        print("And your number is NOT divisible by two!")

else:

    print("You typed a Zero!")





#Lists

#New data type.  Index is the location in the list.  Start counting with zero.

pets=['cat','dog','lizard','snake','hamster','penguins'] 

print(pets[0]) #prints cat.

print(pets[0:3]) #prints a slice of the list from element 0 to 3 but does not include 3

                 #prints ['cat', 'dog', 'lizard']

print(pets[0:5:2]) # prints a slice of the list from element 0 to element 5 counting by 2

                   # prints ['cat', 'lizard', 'hamster']



print(len(pets)) #len() refers to length and returns how many elements are in the list    

                 #prints 6



print(pets[len(pets)-1]) #prints the LAST element of pets, penguins



print(pets.index('snake')) #prints the index/location of snake, 3



grades=[92,60,60,77,95,20,50,84]

print(max(grades))#prints the biggest value in the list, 95

print(max(pets))#prints last element when in alphabetical order

                                          

jake=grades.remove(20)#removes the minimum grade,20, but doesn't assign to jake

carl=grades.pop()#removes the last element,84, and assigns it to carl

print(grades)#prints [92, 60, 60, 77, 95, 50]

print("Jake has "+str(jake)+" and Carl has "+str(carl))

pets.append('frog')#adds frog to the pets list at the end

print(pets)#['cat', 'dog', 'lizard', 'snake', 'hamster', 'penguins', 'frog']



#Finding the max grade of a student with coordinating lists 

students=['john','pepe','pablo','mauricio','michael','maya'] #creates a student list

numbers=[1,34,6,55,20,8] #creates a list of grades associated with each student

biggie=max(numbers) #finds the biggest value

location=numbers.index(biggie) #finds where the biggest value happens

print(students[location])#prints mauricio







#while looping!!!

#repeat the indented section after the while until a condition is met

print("\nWhile loop 1: runs a set number of times:")

someLoopCounter=0

while someLoopCounter<10:

    print("The counter variable has a value of "+ str(someLoopCounter))

    someLoopCounter=someLoopCounter+1

"""This loop prints out the following:

The counter variable has a value of 0

The counter variable has a value of 1

The counter variable has a value of 2

The counter variable has a value of 3

The counter variable has a value of 4

The counter variable has a value of 5

The counter variable has a value of 6

The counter variable has a value of 7

The counter variable has a value of 8

The counter variable has a value of 9"""



print("\nWhile loop 2 version a: depends on user input")

choice=input('Would you like to see this line again? y or n: ')

while choice=='y':

    choice=input('Would you like to see this line again? y or n: ')

#This loop repeats as long as the user types y. This loop stops repeating when the user does not type y.



print("\nWhile loop 2 version b: depends on user input")    

altchoice=input('Would you like to see this line again? y or n: ')

while altchoice!='n':

    altchoice=input('Would you like to see this line again? y or n: ')

#This loop repeats as long as the user does not type n. This loop stops repeating when the user types n


