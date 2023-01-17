tuple1 = (1, 2, 3, 4, 2, 3, 2)
print(tuple1[2])
print(tuple1[-1])
print(tuple1[::-1])     # reverse a tuple
print(tuple1.count(2))  # number of occurrence of 2 in tuple
print(tuple1.index(3))  # first occurrence of 3 in tuple
dict1 = {"name": "ilker", 'age': 33, '''sex''': '''male'''}
print(dict1)
print(dict1['name'])
dict1["nationality"] = "tr"
print(dict1)
print(dict1.keys())
print(dict1.values())
print(dict1.items())
for k, v in dict1.items():
    print(k, v)