# Exercise 3 
# This program exmamines variables x, y, and z and prints the largest odd number among them

#Ask the user to input each number they are interested in finding out which is odd 
x = int(input("Input x: "))
y = int(input("Input y: "))
z = int(input("Input z: "))

#Using a Modulus operator to differentiate which input is odd then figuring out if it is the largest number. Doing this for each input. 
if x%2 != 0 and x > y and y >z:                                    #
    print("x is the largest odd number")
elif y%2 != 0 and y > z and z > x:
    print("y is the largest odd number")
elif z%2 != 0 and z > y and y > x:
    print("z is the largest odd number")
elif x%2 == 0 and y%2 == 0 and z%2 == 0: #Making sure that all numbers are even before the output tells the user there are no odd numbers 
    print("None of the numbers you entered are odd")
#Exercise 4 

numXs = int(input('How many times should I print the letter X? '))
toPrint = numXs * "X"
while True:
    print(toPrint)
    break