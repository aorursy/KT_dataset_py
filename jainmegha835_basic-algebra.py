import numpy as np

u=np.array([3,4])

v=np.array([30,40])
print(np.linalg.norm(u))

print(np.linalg.norm(v))
def direction(x):

    return x/np.linalg.norm(x)

    
w=direction(u)

y=direction(v)

print(w)

print(y)
print(np.linalg.norm(w))

print(np.linalg.norm(y))
import math



def geometric_dot_product(x,y,theta):

    x_norm=np.linalg.norm(x)

    y_norm=np.linalg.norm(y)

    return x_norm*y_norm*math.cos(math.radians(theta))
theta=60

# resuce theta ,dot prudct will increase

X=[3,5]

y=[8,2]



print(geometric_dot_product(X,y,theta))
def dot_product(X,y):

    result=0

    for i in range(len(X)):

        result=result+X[i]*y[i]

    return(result)
X=[3,5]

y=[8,2]

print(dot_product(X,y))

# without theta