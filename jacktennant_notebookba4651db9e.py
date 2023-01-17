print("Hello World")
# This is a comment

print("Hello, World!")
x = 5 #Integer

y = 10 #Integer

word = "Wales" #String

decimalNumber = 2.15 #Float

booleanTest = True #Boolean
print(x)

print(word)

print(decimalNumber)

print(booleanTest)
print("x variable 5 is",(type(x)))

print("name variable Wales is",(type(word)))

print("decimalNumber 2.15 variable is",(type(decimalNumber)))

print("booleanTest variable True is",(type(booleanTest)))
#Python can actually combine string data

print ("5" + "5")



#You can use any of the operators "+". "-". "*" "/" for example.

print( 50 + 55)



#Earlier we set x to 5 and y to 10

print (x + y)
#So how do we write a list?

thislist = ["apple", "banana", "cherry", 1]

print(thislist)
#Using square brackets you can retrive data using indexing

print(thislist[1])
print(thislist[0])
print(thislist[0:2]) # a quirk with this is that it retreves the index 0 (0 = apple) but ends at index 2, if you leave the second number blank it will select all values
print(thislist[0:])
thisdict = {

  "brand": "Ford",

  "model": "Mustang",

  "year": 1964

}

print(thisdict)
x = thisdict["model"]

print(x)
fruits = ["apple", "banana", "cherry"]

for items in fruits: # for every item in the fruits list, print out each item

  print(items)
i = 1

while i < 6: # while i is less than 6, print i

  print(i)

  i = i + 1
xtest = 20

ytest = 20



if xtest > ytest:

    print("The value of X is greater than Y")

elif xtest == ytest: # one thing to note is that the "=" is used to assign variables, to check if something is equal Python uses "=="

    print("The value of X is equal to Y")

else:

    print("The value of X is lower than Y")



def my_function():

    print("Hello, this is the output from the function") # i've included variables used above

    print(xtest + ytest)

    print(xtest - ytest)

    print(xtest / ytest)

    print(xtest * ytest)

    print("This is the last statement for the function")



def function_test2():

    print("this will not be included")



  
my_function()