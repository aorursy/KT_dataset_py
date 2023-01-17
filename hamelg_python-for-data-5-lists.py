my_list = ["Lesson", 5, "Is Fun?", True]



print(my_list)
second_list = list("Life is Study")  # Create a list from a string



print(second_list)
empty_list = []



print( empty_list )
empty_list.append("I'm no longer empty!")



print(empty_list)
my_list.remove(5)



print(my_list)
combined_list = my_list + empty_list



print(combined_list)
combined_list = my_list



combined_list.extend(empty_list)



print(combined_list)
num_list = [1, 3, 5, 7, 9]

print( len(num_list))                # Check the length

print( max(num_list))                # Check the max

print( min(num_list))                # Check the min

print( sum(num_list))                # Check the sum

print( sum(num_list)/len(num_list))  # Check the mean*
1 in num_list
1 not in num_list
num_list.count(3)
new_list = [1, 5, 4, 2, 3, 6]      # Make a new list



new_list.reverse()                 # Reverse the list

print("Reversed list", new_list)



new_list.sort()                    # Sort the list

print("Sorted list", new_list)
another_list = ["Hello","my", "bestest", "old", "friend."]



print (another_list[0])

print (another_list[2])
print (another_list[-1])

print (another_list[-3])
print (another_list[5])
nested_list = [[1,2,3],[4,5,6],[7,8,9]]



print (nested_list[0][2])
my_slice =  another_list[1:3]   # Slice index 1 and 2

print(my_slice )
# Slice the entire list but use step size 2 to get every other item:



my_slice =  another_list[0:6:2] 

print(my_slice )
slice1 = another_list[:4]   # Slice everything up to index 4

print(slice1)


slice2 = another_list[3:]   # Slice everything from index 3 to the end

print(slice2)
# Take a slice starting at index 4, backward to index 2



my_slice =  another_list[4:2:-1] 

print(my_slice )
my_slice =  another_list[:]   # This slice operation copies the list

print(my_slice)
my_slice =  another_list[::-1] # This slice operation reverses the list

print(my_slice)
another_list[3] = "new"   # Set the value at index 3 to "new"



print(another_list)



del(another_list[3])      # Delete the item at index 3



print(another_list)
next_item = another_list.pop()



print(next_item)

print(another_list)
list1 = [1,2,3]                        # Make a list



list2 = list1.copy()                   # Copy the list



list1.append(4)                        # Add an item to list 1



print("List1:", list1)                 # Print both lists

print("List2:", list2)
list1 = [1,2,3]                        # Make a list



list2 = ["List within a list", list1]  # Nest it in another list



list3 = list2.copy()                   # Shallow copy list2



print("Before appending to list1:")

print("List2:", list2)

print("List3:", list3, "\n")



list1.append(4)                        # Add an item to list1

print("After appending to list1:")

print("List2:", list2)

print("List3:", list3)
import copy                            # Load the copy module



list1 = [1,2,3]                        # Make a list



list2 = ["List within a list", list1]  # Nest it in another list



list3 = copy.deepcopy(list2)           # Deep copy list2



print("Before appending to list1:")

print("List2:", list2)

print("List3:", list3, "\n")



list1.append(4)                        # Add an item to list1

print("After appending to list1:")

print("List2:", list2)

print("List3:", list3)