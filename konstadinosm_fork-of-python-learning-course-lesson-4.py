var = "datacamp"
print(var)
my_list = list(var)
print(my_list)
x = set(my_list)
print(x)
y = set("dataset")
print(y)
x - y
x | y
x & y
w = tuple("cake")
print(w)
# w[1] = "o"
mydict = {'Clara': 123123, 'Anna': 456456, 'Monica': 789789, 'Eleonora': 100001}
print(mydict)
len(mydict)
mydict.keys()
mydict.values()
mydict['Anna']
del mydict['Anna']
print(mydict)
mydict['Monica'] = 321321
print(mydict)
print(type(var))
print(type(my_list))
print(type(x))
print(type(w))
print(type(mydict))
mydict = {'Clara': 123123, 'Anna': 456456, 'Monica': 789789, 'Eleonora': 100001}
print(mydict)
mydict2 = mydict
print(mydict2)
del mydict2['Anna']
print(mydict)
print(mydict2)
mydict = {'Clara': 123123, 'Anna': 456456, 'Monica': 789789, 'Eleonora': 100001}
print(mydict)
mydict2 = mydict.copy()
print(mydict2)
del mydict2['Anna']
print(mydict)
print(mydict2)