list_a = ["Apple","Orange","Banana","Grape"]

tuple_a = ("Apple","Orange","Banana","Grape")

dict_a = {1: "Apple",2: "Orange",3: "Banana",4: "Grape"}

#list is mutable in nature, and the slicing operator can be used on it

list_a[1] = "Lemon"
print(list_a)

list_a.insert(4,"Strawberry")
print(list_a)

list_a.pop(3)
print(list_a)

print(list_a[0:2])


#tuples are not mutable in nature, although the slicing operator can be used
# unlike lists,no changes can be made to the tuple
print(tuple_a)

print(tuple_a[0:2])

# tuple_a[2] = "Kiwi", this would lead to an error 


#dictionaries have two parts, i.e keys and values, for eg, dict = {key1:"value1", key2:"value2"}

print(dict_a)

#in dictionary the "keys" are like tuples and are immutable while "values" are like lists and mutable
#to change the values, we might reference the "values" through the "keys"


dict_a[1] = "Lemon"

print(dict_a)

#removing a key and its value

del dict_a[1]
print(dict_a)

#adding a new key and value

dict_a[1] = "Kiwi"

print(dict_a)