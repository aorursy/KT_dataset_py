# Assign a list to an variable named my_list
my_list = [1,2,3]
my_list = ['A string',23,100.232,'o']
len(my_list)
my_list = ['one','two','three',4,5]
# Grab element at index 0
my_list[0]
# Grab index 1 and everything past it
my_list[1:]
# Grab everything UP TO index 3
my_list[:3]
my_list + ['new item']
my_list
# Reassign
my_list = my_list + ['add new item permanently']
my_list
# Make the list double
my_list * 2
# Again doubling not permanent
my_list
# Create a new list
list1 = [1,2,3]
# Append
list1.append('append me!')
# Show
list1
# Pop off the 0 indexed item
list1.pop(0)
# Show
list1
# Assign the popped element, remember default popped index is -1
popped_item = list1.pop()
popped_item
# Show remaining list
list1
list1[100]
new_list = ['a','e','x','b','c']
#Show
new_list
# Use reverse to reverse order (this is permanent!)
new_list.reverse()
new_list
# Use sort to sort the list (in this case alphabetical order, but for numbers it will go ascending)
new_list.sort()
new_list
# Let's make three lists
lst_1=[1,2,3]
lst_2=[4,5,6]
lst_3=[7,8,9]

# Make a list of lists to form a matrix
matrix = [lst_1,lst_2,lst_3]
# Show
matrix
# Grab first item in matrix object
matrix[0]
# Grab first item of the first item in the matrix object
matrix[0][0]
# Build a list comprehension by deconstructing a for loop within a []
first_col = [row[0] for row in matrix]
first_col