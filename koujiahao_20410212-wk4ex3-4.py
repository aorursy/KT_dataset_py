# Exercise 3

x = int(input("Input x: "))
y = int(input("Input y: "))
z = int(input("Input z: "))

print("")

if x % 2 != 0:
    if x > y and x > z:
        print(str(x),"is the largest odd number")
    elif x < z and x > y and y % 2 != 0:
        print(str(x),"is the largest odd number")
    elif x < y and x > z and z % 2 != 0:
        print(str(x),"is the largest odd number")
    else:
        print("")
        
if y % 2 != 0:
    if y > x and y > z:
        print(str(y),"is the largest odd number")
    elif y < x and y > z and z % 2 != 0:
        print(str(y),"is the largest odd number")
    elif y < z and y > x and x % 2 != 0:
        print(str(y),"is the largest odd number")
    else:
        print("")
        
if z % 2 != 0:
    if z > x and z > y:
        print(str(z),"is the largest odd number")
    elif z < x and z > y and y % 2 != 0:
        print(str(z),"is the largest odd number")
    elif z < y and z > x and x % 2 != 0:
        print(str(z),"is the largest odd number")
    else:
        print("")
        
if x % 2 == 0 and y % 2 == 0 and z % 2 == 0:
        print("None of x,y,z is odd number")
# Exercise 4

numXs = int(input("How many times should I print the letter X?"))
x = numXs

while True:
    if x > 0:
        break
    
toPrint = x * 'X'
print(toPrint)