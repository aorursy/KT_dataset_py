#This program examines three variables—x, y, and z—and prints the largest odd
#number among them. If none of them are odd, it print a message to that effect
x = 2
y = 4
z = 6

if x%2 != 0 and x > y and y >z:                                    #
    print("x is the largest odd among x, y, and z")
elif y%2 != 0 and y > z and z > x:
    print("y is the largest odd number among x, y, and z")
elif z%2 != 0 and z > y and y > x:
    print("z is the largest odd number among x, y, and z")
elif x%2 == 0 or y%2 == 0 or z%2 == 0:
    print("None of them are odd!")
#Replace the comment in the following code with a while loop.
numXs = int(input('How many times should I print the letter X? '))
toPrint = ""
while True:
    print(toPrint)
    break
