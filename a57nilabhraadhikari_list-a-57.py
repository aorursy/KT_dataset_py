#Creating a list
colors=['Red','Green','Blue']
colors
#Accessing a list element
print(colors[1])
#Reassigning an element in the list
colors[1]='Yellow'
colors
#Inserting an element
colors.insert(1,'Orange')
colors
#Searching and Returning the position in the list
element='Orange'
print(colors.index(element))
    
#Deleting an element in the list
del colors[3]
colors