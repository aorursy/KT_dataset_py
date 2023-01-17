# example: ("*" * x)
#
#fa(1)
#
#fb(1)
# example " " * z + "*" * x
def mybranch(z,x):
    print("-" * z + "*" * x)
    return
mybranch(5,3)
def branchloop():
    spaces = 0
    asterisks = -1
    for k in range(0, 9):
        spaces = 9 - k
        asterisks = asterisks + 2
        mybranch(spaces,  asterisks)
    k = k + 1
    return k
branchloop()
def trunkloop():
    for t in range(0,2):
        mybranch(9, 1)
    return
trunkloop()
def isTrunk(k):
    if k >= 9:
        trunkloop()
    return
isTrunk(9)
def mytree():
    var = branchloop()
    isTrunk(var)
    return
mytree()
