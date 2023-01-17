# Very notebook, such code.  Wow

def add(a, b):
    return a + b

def fact(a):
    if a < 1:
        return 0
    if a == 1:
        return 1
    return a * fact(a-1)

def cool_new_function(name):
    return "Hello {}".format(name)

myvar = 5
othervar = 100
print(fact(myvar))
print(add(myvar, othervar))
print(cool_new_function("Jim"))