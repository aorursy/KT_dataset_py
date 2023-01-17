#Creating a Tupple
percentages=(99,95,90,89,93,96)
percentages
#Insering elements in the tupple
element=(78,87)
percentages=percentages+element
percentages

# Search an element in tuple
percentages.index(78)
#Replace and Deletion of an element is not possible in tuple because it is immutable
#But Deletion of a tuple is possible
del percentages
percentages