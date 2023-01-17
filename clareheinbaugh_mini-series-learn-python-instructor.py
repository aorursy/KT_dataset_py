# Create a string, integer, float, and boolean
my_string = 'Hello world!'
print(my_string)
my_integer = 5
print(my_integer)
my_float = 5.0
print(my_float)
my_boolean = False
print(my_boolean)
# List (duplicate items allowed; order is preserved)
grocery_list = ['apple', 'banana', 'pear', 'pear']
print(grocery_list)
grocery_list.append('squash')
print(grocery_list)
second_element = grocery_list[1]
print(second_element)
# For-loop
for grocery_item in grocery_list:
    print(grocery_item)
# Set (unique items; order changes)
grocery_set = set(grocery_list)
print(grocery_set)

if 'squash' in grocery_list:
    print('Yay!')
else:
    print('Boo!')
# Dictionary
grocery_dictionary = {'pear': 2, 'banana': 3, 'apple': 1, 'squash': 1}
print(grocery_dictionary)
number_of_bananas = grocery_dictionary['banana']
print(number_of_bananas)
grocery_dictionary['kiwi'] = 5
print(grocery_dictionary)