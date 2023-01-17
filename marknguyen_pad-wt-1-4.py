# Save sequences of data



l1 = [1,2,3,4,5,6,7,8,9,10]

t1 = (3,6,9,12,15,18,21,24,27,30)
## Print first three elements of each data structure

## Print last three elements of each data structure

## Print every second element

## Reverse the elements in each data structure

## Print the first and last element of each data structure

## Change the last three elements of l1 to be the first three elements of t1

## Append elements to an existing list

# Alternative ways to add elements to an existing list

print(l1 + [14,15,16])   # Creates a new list

l1.extend([17,18,19])    # Modify the existing list

print(l1)
## Change the last element of t1 to be the first elements of l1 (You should get an error)

t1[-1] = l1[0]