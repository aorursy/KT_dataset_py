from tabulate import tabulate
#inputs
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = [1, 1, 1, 1]
w2 = [1, 1, 1, 1]
t = 2
#output
print("x1    x2    w1   w2     t     O")
for i in range(len(x1)):
    if ( x1[i]*w1[i] + x2[i]*w2[i] ) >= t:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',t,'   ', 1)
    else:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',t,'   ', 0)
#inputs
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = [1, 1, 1, 1]
w2 = [1, 1, 1, 1]
t = 1
#output
print("x1    x2    w1   w2     t     O")
for i in range(len(x1)):
    if ( x1[i]*w1[i] + x2[i]*w2[i] ) >= t:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',t,'   ', 1)
    else:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',t,'   ', 0)
#inputs
x = [0, 1]
w = [-1, -1]
t = 0
#output
print("x      w     t     O")
for i in range(len(x)):
    if ( x[i]*w[i]) >= t:
        print(x[i],'   ',w[i],'   ',t,'   ', 1)
    else:
        print(x[i],'   ',w[i],'   ',t,'   ', 0)
#inputs
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = [-1, -1, -1, -1]
w2 = [-1, -1, -1, -1]
t = -2
#output
print("x1    x2    w1     w2      t     O")
for i in range(len(x1)):
    if ( x1[i]*w1[i] + x2[i]*w2[i] ) > t:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',t,'   ', 1)
    else:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',t,'   ', 0)
#inputs
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = [1, 1, 1, 1]
w2 = [1, 1, 1, 1]
t = 0
#output
print("x1    x2    w1   w2     t     O")
for i in range(len(x1)):
    if ( x1[i]*w1[i] + x2[i]*w2[i] ) <= t:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',t,'   ', 1)
    else:
        print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',t,'   ', 0)
#inputs
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = [1, 1, 1, 1]
w2 = [1, 1, 1, 1]
w3 = [1, 1, 1, 1]
w4 = [-1, -1, -1, -1]
w5 = [-1, -1, -1, -1]
w6 = [1, 1, 1, 1]
t1 = [0.5,0.5,0.5,0.5]
t2 = [-1.5,-1.5,-1.5,-1.5]
t3 = [1.5,1.5,1.5,1.5]
def XOR (a, b):
    if a != b:
        return 1
    else:
        return 0
#output
print('x1    x2    w1    w2    w3     w4    w5     w6    t1        t2      t3     O')
for i in range(len(x1)):
    print(x1[i],'   ',x2[i],'   ',w1[i],'   ',w2[i],'   ',w3[i],'   ',w4[i],'   ',w5[i],'   ',w6[i],'   ',t1[i],'   ',t2[i],'   ',t3[i],'   ',XOR(x1[i],x2[i]))
