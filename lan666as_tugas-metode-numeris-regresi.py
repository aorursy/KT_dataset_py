import numpy as np

import matplotlib.pyplot as plt



smooth = 20
x = np.array([1.0, 2.0, 2.5, 4.0, 6.0, 8.0, 8.5])

y = np.array([0.4, 0.7, 0.8, 1.0, 1.2, 1.3, 1.4])
plt.figure(figsize=(12,9))

plt.scatter(x, y, c='#0000FF')

plt.grid()

plt.xlabel('x')

plt.ylabel('y') 

plt.show()
n = len(x)

def regresiLinear(xr,yr):

    xryr = xr*yr

    xr2 = xr*xr

    br = (n*xryr.sum()-xr.sum()*yr.sum())/(n*xr2.sum()-xr.sum()**2)

    ar = (np.mean(yr) - br*np.mean(xr))

    return ar, br



a1, b1 = regresiLinear(x,y)

print("a = {}\nb = {}".format(a1,b1))
s1 = "y = {:.7f} + {:.7f}x".format(a1, b1)

print("Persamaan Linear (LSE): y = {}".format(s1))
def eval1(l):

    return (a1+b1*l)



def MSE(yPredr):

    yDtr = (y-y.mean())**2

    yDr = (y - yPredr)**2

    return (yDr.sum()/n)



yPred1 = eval1(x)

modelMSE1 = MSE(yPred1)

baseMSE = MSE(y.mean())

print("Mean Squared Error (MSE) = {}".format(modelMSE1))
def RMSE(r):

    return np.sqrt(r)



modelRMSE1 = RMSE(modelMSE1)

print("Root Mean Squared Error (RMSE) = {}".format(modelRMSE1))
def MAE(yPredr):

    dAbsr = np.absolute(y - yPredr)

    return (dAbsr.sum()/n)



modelMAE1 = MAE(yPred1)

print("Mean Absolute Error (MAE)) = {}".format(modelMAE1))
def rR(MSEmr):

    Rr = (1 - (MSEmr/baseMSE))

    rr = np.sqrt(Rr)

    return rr, Rr



r1, R1 = rR(modelMSE1)

print("Koefisien Korelasi (r) = {}\nDerajat Kesesuaian (R) = {}".format(r1, R1))
def plot(s,f):

    xreg = np.array(np.arange(int(x[0]-1)*smooth,int(x[-1]+1)*smooth))

    xreg = xreg/smooth

    print("x regresi: {}".format(xreg))

    yreg = (f(xreg))

    print("y regresi: {}".format(yreg))

    plt.figure(figsize=(12,9))

    plt.scatter(x, y, c="#0000FF", label="Data")

    plt.plot(xreg, yreg, "r", label=s)

    plt.legend(loc="lower right")

    plt.grid()

    plt.xlabel('x')

    plt.ylabel('y')

    

plot(s1, eval1)
p = np.log(x)

q = np.log(y)



A2, B2 = regresiLinear(p,q)

print("A = {}\nB = {}".format(A2, B2))
a2 = np.exp(A2)

b2 = B2

s2 = "{:.7f}x^({:.7f})".format(a2,b2)

print("Persamaan Berpangkat: y = {}".format(s2))
def eval2(l):

    return (a2*(l**b2))



yPred2 = eval2(x)

modelMSE2 = MSE(yPred2)

print("Mean Squared Error (MSE) = {}".format(modelMSE2))
modelRMSE2 = RMSE(modelMSE2)

print("Root Mean Squared Error (RMSE) = {}".format(modelRMSE2))
modelMAE2 = MAE(yPred2)

print("Mean Absolute Error (MAE) = {}".format(modelMAE2))
r2, R2 = rR(modelMSE2)

print("Koefisien Korelasi (r) = {}\nDerajat Kesesuaian (R) = {}".format(r2, R2))
plot(s2, eval2)
v = x

w = np.log(y)



A3, B3 = regresiLinear(v, w)

print("A = {}\nB = {}".format(A3, B3))
a3 = np.exp(A3)

b3 = B3

s3 = "{:.7f}e^({:.7f}x)".format(a3,b3)

print("Persamaan Eksponensial: y = {}".format(s3))
def eval3(l):

    return (a3 * (np.e**(l*b3)))



yPred3 = eval3(x)

modelMSE3 = MSE(yPred3)

print("Mean Squared Error (MSE) = {}".format(modelMSE3))
modelRMSE3 = RMSE(modelMSE3)

print("Root Mean Squared Error (RMSE) = {}".format(modelRMSE3))
modelMAE3 = MAE(yPred3)

print("Mean Absolute Error (MAE) = {}".format(modelMAE3))
r3, R3 = rR(modelMSE3)

print("Koefisien Korelasi (r) = {}\nDerajat Kesesuaian (R) = {}".format(r3, R3))
plot(s3, eval3)
orde = 3



def ai(i):

    return [np.sum(x**j) for j in range(i,i+orde+1)]



A4 = np.matrix([ai(i) for i in range(0,orde+1)])

print("Matriks A:")

print(A4)
B4 = np.array([np.sum((x**i)*y) for i in range(0,orde+1)])

print("Matriks B:")

print(B4)
X4 = np.linalg.inv(A4).dot(B4)

print("Didapatkan Matriks X:")

print(X4)
print("Persamaan Polinomial orde-{}: y = ".format(orde), end="")

s4 = ""

for i,j in enumerate(np.nditer(X4)):

    if(i==0):

        now = "{:.7f}".format(j) 

        print(now, end="")

        s4 += now

    else:

        now = " + {:.7f}x^{}".format(j,i)

        print(now, end="")

        s4 += now
def eval4(l, coeff):

    result = coeff[-1]

    for i in range(-2, -len(coeff)-1, -1):

        result = result*l + coeff[i]

    return result



yPred4 = np.array([eval4(i,X4.tolist()[0]) for i in x.tolist()])

modelMSE4 = MSE(yPred4)

print("Mean Squared Error (MSE) = {}".format(modelMSE4))
modelRMSE4 = RMSE(modelMSE4)

print("Root Mean Squared Error (RMSE) = {}".format(modelRMSE4))
modelMAE4 = MAE(yPred4)

print("Mean Absolute Error (MAE) = {}".format(modelMAE4))
r4, R4 = rR(modelMSE4)

print("Koefisien Korelasi (r) = {}\nDerajat Kesesuaian (R) = {}".format(r4, R4))
xreg4 = np.array(range(int(x[0]-1)*smooth,int(x[-1]+1)*smooth))

xreg4 = xreg4/smooth

print("x regresi: {}".format(xreg4))

yreg4 = np.array([eval4(i,X4.tolist()[0]) for i in xreg4.tolist()])

print("y regresi: {}".format(yreg4))
plt.figure(figsize=(12,9))

plt.scatter(x, y, c="#0000FF", label="Data")

plt.plot(xreg4, yreg4, "r", label="y = " + s4)

plt.legend(loc="lower right")

plt.grid()

plt.xlabel('x')

plt.ylabel('y')