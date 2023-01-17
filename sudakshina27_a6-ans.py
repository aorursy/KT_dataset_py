#Square brackets [] are lists. The contents in a list are mutable.
sample_list=[1,4,5,7,8,3]

sample_list.insert(0,"Hello")

sample_list.insert(1,"zero")

print(sample_list)
vowels = ['a', 'e', 'i', 'o', 'u']

print(vowels [int (int (3 * 2) / 11)])
print(vowels [-1])
print(vowels  [-1:] )
bacon = [3.14, 'cat', 11, 'cat', True]

print(bacon.index('cat'))
bacon.append(99)

print(bacon)
bacon.remove('cat')

print(bacon)
# append() function adds only one element at the end of the list

# insert() function adds one element at the specified index in the list

a = [1,2,3,4,5]

a.append("Apple")

a.insert(3,"Mango")

print(a)
# pop() function removes the last element from the list, or from the specified index if mentioned

# pop() also returns the index value from which the element was removed

# remove() function removes a given element from the list

# remove() returns no value

print("Before removing anything")

print(a)

print("After pop()")

print(a.pop())

print(a)

print("After remove()")

a.remove('Mango')

print(a)
# list - mutable

# tuple - immutable
spam = ['Alice', 'ants', 'Bob', 'badgers', 'Carol', 'cats']

spam.sort()

print(spam)
spam = ['Alice', 'ants', 'Bob', 'badgers', 'Carol', 'cats']

if 'Bob' in spam:

    print("Present")

else:

    print("Absent")
cpy = spam.copy()

print(cpy)
h = ['a','b','c','d','e']

g = ""

print(g.join(h))