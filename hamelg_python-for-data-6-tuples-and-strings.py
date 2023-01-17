my_tuple = (1,3,5)



print(my_tuple)
my_list = [2,3,1,4]



another_tuple = tuple(my_list)



another_tuple
another_tuple[2]     # You can index into tuples
another_tuple[2:4]   # You can slice tuples
# You can use common sequence functions on tuples:



print( len(another_tuple))   

print( min(another_tuple))  

print( max(another_tuple))  

print( sum(another_tuple))  
another_tuple.append(1)    # You can't append to a tuple
del another_tuple[1]      # You can't delete from a tuple
sorted(another_tuple)
list1 = [1,2,3]



tuple1 = ("Tuples are Immutable", list1)



tuple2 = tuple1[:]                       # Make a shallow copy



list1.append("But lists are mutable")



print( tuple2 )                          # Print the copy
import copy



list1 = [1,2,3]



tuple1 = ("Tuples are Immutable", list1)



tuple2 = copy.deepcopy(tuple1)           # Make a deep copy



list1.append("But lists are mutable")



print( tuple2 )                          # Print the copy
my_string = "Hello world"



my_string[3]    # Get the character at index 3


my_string[3:]   # Slice from the third index to the end


my_string[::-1]  # Reverse the string
len(my_string)
my_string.count("l")  # Count the l's in the string
# str.lower()     



my_string.lower()   # Make all characters lowercase
# str.upper()     



my_string.upper()   # Make all characters uppercase
# str.title()



my_string.title()   # Make the first letter of each word uppercase
my_string.find("W")
my_string.find("w")
my_string.replace("world",    # Substring to replace

                  "friend")   # New substring
my_string.split()     # str.split() splits on spaces by default
my_string.split("l")  # Supply a substring to split on other values
multiline_string = """I am

a multiline 

string!

"""



multiline_string.splitlines()
# str.strip() removes whitespace by default



"    strip white space!   ".strip() 
"xXxxBuyNOWxxXx".strip("xX")
"Hello " + "World"
" ".join(["Hello", "World!", "Join", "Me!"])
name = "Joe"

age = 10

city = "Paris"



"My name is " + name + " I am " + str(age) + " and I live in " + "Paris"
template_string = "My name is {} I am {} and I live in {}"



template_string.format(name, age, city)
# Remaking the example above using an f-string



f"My name is {name} I am {age} and I live in {city}"