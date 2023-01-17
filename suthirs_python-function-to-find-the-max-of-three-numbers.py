x = float(input())

y = float(input())

z = float(input())

# if x is greater than or equal to y / if x is greater than or equal to z, print - x

if (x>=y) and (x>=z):

    print("x is the largest number")

# if y is greater than or equal to x / if y is greater than or equal to z, print - y

elif (y>=x) and (y>=z):

    print("y is the largest number")

# if both conditions are false, print - Z

else:

    print("z is the largest number")