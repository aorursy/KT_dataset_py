import numpy as np
w=np.array([1,-1,0,0.5])
x1=np.array([1,-2,0,-1])
x2=np.array([0,1.5,-0.5,-1])
x3=np.array([-1,1,0.5,-1])
error = 1
c=0.5
d1 = -1
d2 = -1
d3 = 1
i = 0
while(error!=0):
    o = np.sign(np.dot(w,x1))
    e1 = d1 - o
    print('e1=',e1)
    w=w+c*e1*x1
    o=np.sign(np.dot(w,x2))
    e2= d2 - o
    print('e2=',e2)
    w=w+c*e2*x2
    o=np.sign(np.dot(w,x3))
    e3= d3 - o
    print('e3=',e3)
    w=w+c*e3*x3
    error = abs(e1) + abs(e2) + abs(e3)
    print('eroor =',error)
    i += 1
print('Updated weights after', i, 'iterations are:',w)
