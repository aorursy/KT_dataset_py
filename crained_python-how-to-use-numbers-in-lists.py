# let's creates a range of numbers

for value in range(1, 5):

    print(value)
# We want 5 too

for value in range(1, 6):

    print(value)
number_list = list(range(1, 6))

print(number_list)
# We can get just even numbers

even = list(range(2, 11, 2))

print(even)
# Now we want square root. ** is how we get the exponent

squares = []

for value in range(1, 11):

    square = value ** 2

    squares.append(square)

print(squares)
# you can do squares in a short amount of code but sometimes it is easier to 

# have more code if it is easier to read and debug

squares = []

for value in range(1, 11):

    squares.append(value ** 2)

    

print(squares)
# we can do the above code with list comprehension

squares = [value**2 for value in range(1, 11)]

print(squares)
# find the minimum

min(squares)
max(squares)
sum(squares)