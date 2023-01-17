print("$" * 5)
def fa(x):
    print("*" * x)
    return
fa(10)
def fb(z):
    print("-" * z)
    return
fb(3)
# example " " * z + "*" * x
def mybranch(z, x):
    print("-" * z + "*" * x)
    return
mybranch(5,3)
def branchloop(y,w):
    for k in range(0, 9):
        mybranch(y,w)
    return
branchloop(6,8)
def trunkloop():
    for j in range(0,2):
        mybranch(3,7)
    return
trunkloop()
def isTrunk(a):
    if a >= 9:
        trunkloop()
    return
isTrunk(9)
isTrunk(7)