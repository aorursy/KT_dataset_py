list1 = [1, 2, 3, 4, 5]
list2 = ["a", "b", "c", "d"]
list(map(lambda a: a * 2, list1))  # multiply every member of list with 2
list(filter(lambda a: a % 2 == 0, list1))  # filter even numbers
from functools import reduce

reduce(lambda a, b: a + b, list1)  # get total
print(list1)
print(list2)

print(list(zip(list1, list2)))  # merge elements of 2 lists, by index
list(enumerate(list2))  # enumeration of list, sets index to each item
for x in enumerate(list2):
    if x[0] % 2 == 0:
        continue
    print(x[1])
print(any(x > 2 for x in list1))

print(all(x > 2 for x in list1))