#Notice the colour change for Boolean type



x = True

y = False 

print(x)

print(y)

print(type(x))
# Your code goes here try some comparison operations

x = 3

print(x == 3)



print(x > 6)



y = 45

print(y != 45)



print(6.0 == 6)



print('6' == 6)
# Your code goes here

x = True

y = False 

print(x and y)



x = True

y = False

print(x or y)



y = True



print(x and y)



print(not x)



x = 5

y = 6

print(x == 7 or y < 2)
# Your code goes here try different values of x and see what you get

x = 5

#x = - 2

#x = "Howdy"



if x == 0:

    print(x, "is zero")

elif x > 0:

    print(x, "is positive")

elif x < 0:

    print(x, "is negative")

else:

    print(x, "is unlike anything I've ever seen...")
# Your code goes here 



x = 35



if(x % 3 == 0 and x % 5 == 0):

    print("FizzBuzz")

elif (x % 3 ==0):

    print("Fizz")

elif (x % 5 ==0):

    print("Buzz")

else:

    print("Not a multiple of neither 3 nor 5")