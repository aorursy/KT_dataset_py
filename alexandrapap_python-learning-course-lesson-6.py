# example: ("*" * x)
def fa(x):
    print("*" * x)
    return
fa(1)
def fb(z):
    print("-" * z)
    return
fb(3)
# example " " * z + "*" * x
def mybranch(z, x):
    print(" " * z + "*" * x)
    return
mybranch(5,3)
def branchloop():
    for i in range(0, 9):
        print(mybranch(i, i))
    return
def branchloop(y, w):
    for k in range(0, 9):
        mybranch(y, w)
    return
branchloop(5,5)
def trunkloop(y):
    for j in range(0, 2):
        mybranch(y - 1, 1)
    return
trunkloop(9)
def isTrunk(x):
    if x >= 9:
        trunkloop(x)
    return
isTrunk(9)
isTrunk(7)
def branchloop(y):
    spaces = y
    asterisks = -1
    for k in range(0, y):
        spaces = spaces - 1
        asterisks = asterisks + 2
        mybranch(spaces, asterisks)
    k = k + 1
    return k

def isTrunk(k):
    if k >= 9:
        trunkloop(k)
    return
def mytree(y):
    var = branchloop(y)
    isTrunk(var)
    return
mytree(20)