# Create a tuple
t = (1,2,3)
# Check len just like a list
len(t)
# Can also mix object types
t = ('one',2)

# Show
t
# Use indexing just like we did in lists
t[0]
# Slicing just like a list
t[-1]
# Use .index to enter a value and return the index
t.index('one')
# Use .count to count the number of times a value appears
t.count('one')
t[0]= 'change'
t.append('nope')