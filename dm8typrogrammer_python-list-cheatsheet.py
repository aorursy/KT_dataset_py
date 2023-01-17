numbers = [1, 2, 3, 4, 5]
len(numbers)
numbers.append(6)
print(numbers)
print(numbers[0]) # first element
print(numbers[3]) # last element
print(numbers[-1]) # last element
print(numbers[-3]) # 3rd from last
print(numbers[1:3]) 
print(numbers[2:])  # from 3rd element till last
print(numbers[:-2]) # from begining to till second last
print(numbers[:])   # all values
for number in numbers:
    print(number)
numbers.remove(4) # remove 4 from the list 
print(numbers)
del numbers[1] # remove 2nd number
print(numbers)
del numbers[0:3]
print(numbers)
names = ['Alex', 'Jhon', 'Foo', 'Bar']
print(names.pop()) # delete last and return the value
print(names)
names.clear()
print(names)
# finding even number 
numbers = [1, 2, 3, 9, 10]
list(filter(lambda number: (number & 1) == 0, numbers))
numbers = [1, 0, 9, 2, 5, 3]
sorted_numbers = sorted(numbers)
print(numbers) # sorted does not modify list
print(sorted_numbers)
sorted_numbers = sorted(numbers, reverse=True)
print(sorted_numbers)