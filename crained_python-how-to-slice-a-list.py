# let's slice a list to get the first three fruits

fruit = ['apple', 'pear', 'orange', 'berry', 'banana']

print(fruit[0:3])
# let's get 1, 2 and 3 from the list (remember we top before 4th item!)

print(fruit[1:4])
# If you want the first four you can do it in a more condensed way

print(fruit[:4])
# if you want only the last two you can start at the second item in a list

print(fruit[3:])
# you can get the last two items with a negative

print(fruit[-2:])
print("I love these fruits:")

for fruits in fruit[:3]:

    print(fruits.title())
# you can also copy a list using a slice

organic_fruit = fruit[:]

print(organic_fruit)
# add organic fruit

organic_fruit.append('peach')
print(organic_fruit)
print(fruit)