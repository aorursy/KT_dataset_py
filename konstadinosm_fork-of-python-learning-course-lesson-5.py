x = int(input("Please enter an integer: "))
x = x + 10
print(x)
f = lambda x: x+10
print(f(x))
print(f(3))
print(x)
def fun_add(variable):
    variable = variable + 10
    return variable
print(fun_add(x))
fun_add(2)
def add_surname(my_var):
    my_var = my_var + " " + "Chatzisavvidou"
    return my_var
add_surname("Maria")
add_surname("Kostas")
# my_var
mylist = list(range(0,100))
print(mylist[0:4])
mylist[-1]
res1 = []
for k in range(0, len(mylist)):
    res1.append(mylist[k]*10)

print(res1)
res2 = []
res2 = list(map(lambda x: x/10, mylist))
print(res2)
example = []
example = list((lambda x: x/10, mylist))
print(example)
example3 = []
example3 = list(map(lambda x: x/10, mylist[0:5]))
print(example3)
def evenodd(var):
    if var % 2 == 0:
        var_type = "Even"
    else:
        var_type = "Odd"
    return var_type
res3 = []
res3 = list(map(evenodd, mylist))
print(res3)
example2 = []
example2 = list(map(f, mylist))
print(example2)
def myfunction(temp):
    element_type = evenodd(temp)
    if element_type == "Even":
        my_result = temp * 2
    else:
        my_result = temp /10
    return my_result
res4 = []
res4 = list(map(myfunction, mylist))
print(res4)
less_than_50 = []
less_than_50 = list(filter(lambda x: x < 50, res4))
print(less_than_50)
#more_than_50 = []
#more_than_50 = list(filter(x > 50, res4))
#print(more_than_50)
#example10 = list(filter(res4 > 50))
#print(example10)