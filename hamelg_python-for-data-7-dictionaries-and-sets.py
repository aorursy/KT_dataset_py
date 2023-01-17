my_list = [1,2,3,4,5,6,7,8,9,10]



0 in my_list
my_dict = {"name": "Joe",

           "age": 10, 

           "city": "Paris"}



print(my_dict)
my_dict["name"]
my_dict["new_key"] = "new_value"



print(my_dict)
del my_dict["new_key"]



print(my_dict)


len(my_dict)
"name" in my_dict
my_dict.keys()
my_dict.values()
my_dict.items()
my_table_dict = {"name": ["Joe", "Bob", "Harry"],

                 "age": [10,15,20] , 

                 "city": ["Paris", "New York", "Tokyo"]}
my_set = {1,2,3,4,5,6,7}



type(my_set)
my_set.add(8)



my_set
my_set.remove(7)



my_set
6 in my_set
set1 = {1,3,5,6}

set2 = {1,2,3,4}



set1.union(set2)          # Get the union of two sets
set1.intersection(set2)   # Get the intersection of two sets
set1.difference(set2)     # Get the difference between two sets
set1.issubset(set2)       # Check whether set1 is a subset of set2
my_list = [1,2,2,2,3,3,4,5,5,5,6]



set(my_list)