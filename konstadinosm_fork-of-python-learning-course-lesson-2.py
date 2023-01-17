age = 21
name = "Clara"
#print("My name is " + name + " and I am " + age + " years old.")
print(age)
"My name is {} and I am {} years old.".format(name,age)
"My name is {one} and I am {two} years old.".format(two = age, one = name)
print("My name is {one} and I am {two} years old.".format(two = age, one = name))
my_name = "Anna"
my_name[0]
my_name[0:3]
my_name[:]
my_name[-1]
my_name[-2:]
my_set = set(my_name)
my_set
type(my_set)
my_list = [1,2,3,4,5]
my_list
my_list.append(10) #adds an element after the last one, expanding the lenght of the list
my_list
my_list.pop() #removes the last element in the list
my_new_list = my_list.copy()
"""
copies the list to a new list so that changes don't affect the original.
python references lists and changes affect the original list as well.
use copy to explicitly say that you want to create a copy of the original
without any reference or relation.
"""
my_new_list.pop() #this pop only removes the last element in this list, leaving the original as is
print(my_list)
print(my_new_list)
len(my_list)
for k in range(0,len(my_list)):
    my_list[k] = my_list[k] + 100
    print("New element is equal to: " + str(my_list[k]))
my_list
