# Some Opeartions

print("6/2 = \t\t{0}".format(6/2))  # 6.0/2.0 = 3.0

print("6//2 = \t\t{0}".format(6//2)) # 6/2 = 3

print("6*2 = \t\t{0}".format(6*2))  # 6.2 = 12

print("6**2 = \t\t{0}".format(6**2)) # 6^2 = 36

print("9**(1/2) = \t{0}".format(9**(1/2))) # 9^(1/2) = 3.0

print("3 * 'A': \t{0}".format(3 * 'A')) # >>> AAA

# Backslash is wildcard

print("\"\\") # >>> "\
# Python is case sensitive

var = 1

Var = 2

print (var+Var)
#Get Input From Console

#print(input("Your Name: "))
# PreDefined Buit-in Functions:

#type(), len(), float(), str(), int(), round(), del(), sum(), del(), max(), min()

print("float(5): {}".format(float(5)))

print("len(\"AAA\"): {}".format(len("AAA")))

print("int(\"10\"): {}".format(int("10")))

print("str(int(\"10\")): {}".format(str(int("10"))))

print("round(10.3): {}".format(round(10.3)))

print("type(5): {}".format(type(5)))

print("sum([5,4,1]): {}".format(sum([5,4,1])))

print("max([5,4,1]): {}".format(max([5,4,1])))

print("min([5,4,1]): {}".format(min([5,4,1])))

print("del(x): delete x variable")

# Function Defining:

def fun_name(var1, var2 = 5):

    return var1 * var2

fun_name(2, 3)
# Function Overloading:

def fun_name(v1, v2, *args):

    return (v1+v2+args[0]+args[1]+args[2])

print(fun_name(1,1,1,1,1))
# Lambda Functions:

my_fun = lambda v1,v2: v1*v2

print(my_fun(2,3))
# List data type using

list = [[0, 1, 2, 3, 4],

       [10,11,12,13,14],

       [20,21,22,23,24]] # 5x3 list



list
print("list[0]: \t{0}".format(list[0])) # Get first row

print("list[-1]: \t{0}".format(list[-1])) # Get last row

print("list[-2]: \t{0}".format(list[-2])) # Get second to last row

print("list[0][0]: \t{0}".format(list[0][0])) # Get first member in first row

print("list[0:2]: \t{0}".format(list[0:2])) # Get [0->2) members
# Both str and int typed list can be created

list = ["A", 1, "B"]

list
# Get Buit-in Functions.

print("dir(): {}".format(dir()))
# list.methods()

print("dir(list): {}".format(dir(list)))
# int.methods()

print("dir(int): {}".format(dir(int)))
# str.methods()

print("dir(str): {}".format(dir(str)))
# bool.methods()

print("dir(bool): {}".format(dir(bool)))
# list method example

list = [1,3,2,5,4]

print(type(list))

print("list: \t\t\t{}".format(list))

list.append(6)

print("append(6): \t\t{}".format(list))

list.remove(6)

print("remove(6): \t\t{}".format(list))

list.insert(0,0)

print("insert(0,0): \t\t{}".format(list))

list.reverse()

print("reverse(): \t\t{}".format(list))

list.sort()

print("sort(): \t\t{}".format(list))

list.sort(reverse=True)

print("sort(reverse=True): \t{}".format(list))

list.extend("ZAC")

print("extend(\"ZAC\"): \t\t{}".format(list))

list2 = list.copy()

print("list2 = list.copy(): \t{}".format(list2))

list.clear()

print("list.clear(): \t\t{}".format(list))

print("list2.count(\"Z\"): \t{}".format(list2.count("Z")))

print("list2.count(\"0\"): \t{}".format(list2.count("0")))

print("list2: \t\t\t{}".format(list2))

print("list2.index(5): \t{}".format(list2.index(5)))

list2.pop(0)

print("list2.pop(0): \t\t{}".format(list2))
# Get help built-in functions

help(list.insert)

help(list.extend)

help(list.pop)
# Tuple data type:

tuple = (1,2,3,"a","b")

print(type(tuple))

# All build-in functions same as list.

print("{} \n".format(tuple))

print("dir(tuble): {}".format(dir(tuple)))
# Dictionary data type:

dictionary = {"A":1, "B":2, "C":3}

print(type(dictionary))

print(dictionary)

print(dictionary.keys())

print(dictionary.values())

print("\ndir(dictionary): {}".format(dir(dictionary)))
# if statement:

def compare(v1, v2):

    v1 = 1

    v2 = 2

    if (v1>v2):

        return "v1>v2"

    elif (v1==v2):

        return "v1==v2"

    else:

        return "v1<v2"

    

print(compare(1,2))
# if - in statements

dictionary = {"A":1, "B":2, "C":3}

if "A" in dictionary.keys():

    print("dictionary has \"A\"")

elif "a" in dictionary:

    print("dictionary has \"a\"")



# for loop:

for each in range(0,10):

    print(each, end=' ')
for each in "Ali Veli":

    print(each, end='/')
for each in "Ali Veli".split():

    print(each, end='+')
list = [1,2,3]

sum = 0

for each in list:

    sum += each

print(sum)
# while loop:

var = 0

while(var < 10):

    var = var + 1

    print(var, end=' ')