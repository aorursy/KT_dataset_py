# x = int(input("Please enter an integer: "))
#x + 10
#print(x)
#f = lambda x: x+10
#print(f(x))
#def add(variable):
#    variable = variable + 10
#    return variable
#print(add(x))
#mylist = list(range(0,100))
#print(mylist[0:4])
"""
res1 = []
for k in range(0, len(mylist)):
    res1.append(mylist[k]*10)

print(res1)
"""
#res2 = []
#res2 = list(map(lambda x: x/10, mylist))
#print(res2)
"""
def evenodd(var):
    if var % 2 == 0:
        var_type = "Even"
    else:
        var_type = "Odd"
    return var_type
"""
#res3 = []
#res3 = list(map(evenodd, mylist))
#print(res3)
"""
def myfunction(temp):
    element_type = evenodd(temp)
    if element_type == "Even":
        my_result = temp * 2
    else:
        my_result = temp /10
    return my_result
"""
#res4 = []
#res4 = list(map(myfunction, mylist))
#print(res4)
#less_than_50 = []
#less_than_50 = list(filter(lambda x: x < 50, res4))
#print(less_than_50)