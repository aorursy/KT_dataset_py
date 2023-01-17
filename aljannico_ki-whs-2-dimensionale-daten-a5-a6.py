import numpy as np
import matplotlib.pyplot as plt

#  Daten zu Aufgabe 5
X = [10,6,7,5,2,4,3,2,6,8] 
Y = [ 2,4,2,3,5,1,3,4,2,1] 
plt.scatter(X,Y)

x_mean =  np.mean(X)
y_mean =  np.mean(Y)
n = len(X)

print("Mittelwerte für X: ",x_mean, "und Y: ",y_mean)

def kovarianz(X,Y):
    x_mean,y_mean = np.mean(X),np.mean(Y)
    X_diff = [x-x_mean for x in X]
    Y_diff = [y-y_mean for y in Y]
    return sum([x_diff*y_diff for x_diff,y_diff in zip(X_diff,Y_diff)])/len(X)

cov_xy = kovarianz(X,Y)
print("Kovarianz von X und Y: ",cov_xy)
print("Anmerkung: Zähler der Kovarianz: ",cov_xy*n)
import math
def standardabweichung(D):
    mean = np.mean(D)
    return math.sqrt(sum([(d-mean)**2 for d in D])/len(D))
    
sdev_x = standardabweichung(X)    
sdev_y = standardabweichung(Y)    

r_xy = cov_xy/(sdev_x*sdev_y)
print("Standardabweichungen: X =",sdev_x," Y =",sdev_y)
print("Korrelationskoeffizient r=",r_xy)
def varianz(D):
    return standardabweichung(D)**2 # Normalerweise macht man das umgekehrt ;)

var_x = varianz(X)
beta = cov_xy/var_x
alpha = y_mean - beta*x_mean

print("Ausgleichsgerade für Y gegeben X: y =",beta,"* x +",alpha)

def linear_factory(x,m=1,b=0):
    return m*x+b

from functools import partial
regression_xy = np.vectorize(partial(linear_factory,m=beta,b=alpha))

x_show = np.linspace(1,11,5)
y_show = regression_xy(x_show)

plt.scatter(X,Y)
plt.plot(x_show,y_show)
plt.show()
det_xy = r_xy**2
print("Bestimmtheitsmaß:",det_xy)
X =  [0.3, 2.2, 0.5, 0.7, 1.0, 1.8, 3.0, 0.2, 2.3]
Y =  [5.8, 4.4, 6.5, 5.8, 5.6, 5.0, 4.8, 6.0, 6.1]

x_mean =  np.mean(X)
y_mean =  np.mean(Y)

cov_xy = kovarianz(X,Y)
var_x = varianz(X)

beta = cov_xy/var_x
alpha = y_mean - beta*x_mean

print("Ausgleichsgerade für Y gegeben X: y =",beta,"* x +",alpha)
gerade2 = partial(linear_factory,m=beta,b=alpha)
regression2_xy = np.vectorize(gerade2)

x_show = np.linspace(0,3.3,5)
y_show = regression2_xy(x_show)

plt.scatter(X,Y)
plt.plot(x_show,y_show)
plt.show()
var_y = varianz(Y) # Varianz von Y
beta_y = cov_xy/var_y
alpha_y = x_mean - beta_y*y_mean

print("Ausgleichsgerade für X gegeben Y: x =",beta_y,"* y +",alpha_y)

# Umstellen anch y
beta_y_neu = 1/beta_y
alpha_y_neu = -alpha_y/beta_y

print("Nach Y umgestellte Ausgleichsgerade für X gegeben Y: y =",1/beta_y,"* x +",-alpha_y/beta_y)

regression3_xy = np.vectorize(partial(linear_factory,m=beta_y_neu,b=alpha_y_neu))

y_show_neu = regression3_xy(x_show)

plt.scatter(X,Y)
plt.plot(x_show,y_show)
plt.plot(x_show,y_show_neu)
plt.show()
def r_steigungen(m_x,m_y):
    sign = -1 if m_x < 0 else 1
    return sign*math.sqrt(m_x*m_y)

print("Korrelationskoeffizient aus den Steigungen: r=",r_steigungen(beta,beta_y))
corr_xy = cov_xy/(standardabweichung(X)*standardabweichung(Y))
print("Korrelationskoeffizient r=",corr_xy)
def r_quadrat(X,Y,gerade):
    y_mean = np.mean(Y)
    nenner = sum([(y-y_mean)**2 for y in Y])
    zaehler = sum([(y-gerade(x))**2 for x,y in zip(X,Y)])
    return zaehler,nenner,1-zaehler/nenner

SQR,SQT,det_xy = r_quadrat(X,Y,gerade2)
print("SQT =",SQT,"  SQR =",SQR,"  SQE =",SQT-SQR)

print("\nBestimmtheitsmaß R^2 über SQE/SQT:", det_xy)
print("Bestimmtheitsmaß R^2 über r:", corr_xy**2)