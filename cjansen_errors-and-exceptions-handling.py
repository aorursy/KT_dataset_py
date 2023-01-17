1 + "1"
1/0
spglobal(8)
def simpleSum(a, b):
    try:
        c = a + b
        return c
    except:
        print("there was an error")

a = 1
b = "1"
simpleSum(a,b)
def simpleSum(a, b):
    try:
        c = a + b
    except TypeError:
        c = float(a) + float(b)
    return c

a = 1
b = "1"
print(simpleSum(a,b))
