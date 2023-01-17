import numpy as np

from matplotlib import pyplot as plt
w=np.array([-2.5,1.75])

x1=np.array([1,1])

x2=np.array([-0.5,1])

x3=np.array([3,1])

x4=np.array([-2,1])

d1=1

d2=-1

d3=1

d4=-1

iter=0

e=1

c=0.5

k=[]

k.append(w)
while(e!=0):

    o=np.sign(np.dot(w,x1))

    e1=d1-o

    w=w+c*e1*x1

    print('w1=',w)

    k.append(w)

    o=np.sign(np.dot(w,x2))

    e2=d2-o

    w=w+c*e2*x2

    print('w2=',w) 

    k.append(w)

    o=np.sign(np.dot(w,x3))

    e3=d3-o

    w=w+c*e3*x3

    print('w3=',w)

    k.append(w)

    o=np.sign(np.dot(w,x4))

    e4=d4-o

    w=w+c*e4*x4

    print('w4=',w)

    k.append(w)

    e=abs(e1)+abs(e2)+abs(e3)
print('updated iteration=',w)
a1=np.zeros(17)

a2=np.zeros(17)

for i in range(17):

    a1[i]=k[i][0]

    a2[i]=k[i][1]  

o1=np.array([0,0])

class1x = [-5,5]

class1y = [5,-5]

class2x = [-5,5]

class2y = [-3,3]

class3x = [2,-2]

class3y = [-5,5]

class4x = [-3,3]

class4y = [-5,5]
plt.figure(figsize=(9, 9))

plt.plot(a1,a2)    

o1=np.array([0,0])

plt.plot(o1,x1)

plt.plot(class1x,class1y,'g-')

plt.plot(class2x,class2y, 'r-')

plt.plot(class3x,class3y, 'g-')

plt.plot(class4x,class4y, 'r-')

plt.grid(True)

plt.xlabel('W1')

plt.ylabel('W2')

plt.title('Graphical Representaion of Perceptron Learning Rule')

plt.annotate('Class1(WY1)', xy=(-4, 1), xytext=(-4, 4)

             )

plt.annotate('Class1(WY3)', xy=(-1.5, 1), xytext=(-1.5, 4)

             )

plt.annotate('Class2(WY2)', xy=(-4, 1), xytext=(-5, -2)

             )

plt.annotate('Class2(WY4)', xy=(-1.5, 1), xytext=(-1.5, -3)

             )

plt.annotate('W0', xy=(-4, 1), xytext=(-2.5, 2)

             )

plt.annotate('W1', xy=(-1.5, 1), xytext=(-1.5, 3)

             )

plt.annotate('W2', xy=(-4, 1), xytext=(-1, 2)

             )

plt.annotate('W3', xy=(-1.5, 1), xytext=(2, 3)

             )

plt.annotate('W4', xy=(-1.5, 1), xytext=(3, 1)

             )

plt.show()


