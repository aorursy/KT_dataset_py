# Creating a List of 5 numbers

List = [1,2,3,4,5] 

print(List) 

# Accesing elements in a list 

print("Third Element :",List[2])

# Adding elements to the list

List.append(6)

print("\n List after append() :" ,List)

# print the last element of list 

print("\n Last Element of the list :",List[-1]) 

# Removing elements from List using remove()

List.remove(6) 

# Slicing operation in list

List_segment = List[2:5] 

print("\nSlicing elements in a range 2-4: ") 

print(List_segment)
# Creating a Dictionary  

Dict = {1: 'The', 2: 'workshop', 3: 'for'} 

print("\nDictionary with the use of Integer Keys: ") 

print(Dict) 

# Adding elements to the dictionary 

Dict[4] = "WiDS"

print("\nDictionary after adding 4 th element: ") 

print(Dict) 

# accessing a element using key 

print("Accessing a element using key:") 

print(Dict[1]) 

#Creatting a Tuple of 5 numbers

Tuple1 = ('1','2','3','4','5') 

print("Tuple :",Tuple1)

#Accessing Tuple with Indexing 

print("\nFirst element of Tuple: ") 

print(Tuple1[1]) 

# Concatenaton of tuples 

Tuple2 = (0,1,2,3) 

Tuple3 = (4,5,6,7)   

Tuple4 = Tuple1 + Tuple2 

print("Tuple after concatenation :",Tuple4)

# Deleting a Tuple  

Tuple5 = (0, 1, 2, 3, 4) 

del Tuple5   

 
# Creating a String  

String1 = 'This is WiDS workshop'

print("Printing the string: ") 

print(String1)

# Printing First character 

print("\nFirst character of String is: ") 

print(String1[0]) 

# Printing 3rd to 12th character 

print("\nSlicing characters from 3-12: ") 

print(String1[3:12])

# Escaping Sequencing of a String 

String1 = 'This is \"WiDS workshop \"'

print("\nEscape sequencing of a string: ") 

print(String1)  

# Formatting of Strings 

String1 = "{} {} {}".format('This', 'is', 'WiDS Workshop') 

print("\nFormatting of String: ") 

print(String1) 
