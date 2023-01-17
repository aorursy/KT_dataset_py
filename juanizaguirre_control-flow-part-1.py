i = 1

while i < 6:

    print(i)

    i += 1
if x < 0:

    x = 0

    print('Negative changed to zero')

elif x == 0:

    print('Zero')

elif x == 1:

    print('Single')

else:

    print('More')
# Measure some strings:

animals = ['cat', 'dog', 'rabbit', 'tiger', 'lion']

for animal in animals:

    print(animal, len(animal))
my_list = [0, 1, 2, 3, 4]

for elem in my_list:

    #if elem >= 4:

        #my_list.append(len(my_list))

    print("Length is: {}".format(len(my_list)))
animals = ['cat', 'dog', 'rabbit', 'tiger', 'lion']

print("list before loop: {}".format(animals))

# Strategy:  Iterate over a copy

for animal in animals.copy():

    if animal == 'rabbit':

        animals.remove(animal)

print("list after loop: {}".format(animals))
# Strategy:  Create a new list

animals = ['cat', 'dog', 'rabbit', 'tiger', 'lion']

print("list before loop: {}".format(animals))

# Strategy:  Iterate over a copy

for animal in list(animals):

    if animal == 'rabbit':

        animals.remove(animal)

print("list after loop: {}".format(animals))
for n in range(2, 10):

    for x in range(2, n):

        if n % x == 0:

            print(n, 'equals', x, '*', n//x)

            break

    else:

        # loop fell through without finding a factor

        print(n, 'is a prime number')
for num in range(2, 10):

    if num % 2 == 0:

        print("Found an even number", num)

        continue

    print("Found a number", num)