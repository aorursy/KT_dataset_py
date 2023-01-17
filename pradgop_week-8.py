from __future__ import print_function
import scipy.signal as sp
import pylab as p
import numpy as np
from sympy import *
import matplotlib.pyplot as plt

pi=np.pi
s=symbols('s')
def get_coeff(x):
    x=x.simplify()
    num_coeffs,den_coeffs=expand(x).as_numer_denom()
    num_coeff=Poly(num_coeffs,s).all_coeffs()
    den_coeff=Poly(den_coeffs,s).all_coeffs()
    num_coeff=[float(num_coeff[i]) for i in range(len(num_coeff))]
    den_coeff=[float(den_coeff[i]) for i in range(len(den_coeff))]
    return num_coeff,den_coeff
def lowpass(R1,R2,C1,C2,G,Vi): 
    A=Matrix([[0,0,1,-(1/G)],[-1/(1+s*C2*R2),1,0,0],[0,-G,G,1],[((1/R1)+(1/R2)+(s*C1)),(-1/R2),0,(-s*C1)]])
    b=Matrix([0,0,0,(Vi/R1)])
    V=A.inv()*b
    return(A,b,V)

def highpass(R1,R2,C1,C2,G,Vi): 
    A=Matrix([[0,0,1,-(1/G)],[-1/(1+(1/R2)*(1/(s*C2))),1,0,0],[0,-G,G,1],[((s*C1)+(s*C2)+(1/R1)),(-s*C2),0,(-1/R1)]])
    b=Matrix([0,0,0,(Vi*s*C1)])
    V=A.inv()*b
    return(A,b,V)
A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)
num_coeff,den_coeff=get_coeff(V[3])
#print(num_coeff,den_coeff)
H=sp.lti(num_coeff,den_coeff)
w=p.logspace(0,8,801)
ss=1j*w
hf=lambdify(s,V[3],"numpy")
v=hf(ss)
p.loglog(w,abs(v),lw=2)
p.grid(True)
p.show()

A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1)
num_coeff,den_coeff=get_coeff(V[3])
#print(num_coeff,den_coeff)
H=sp.lti(num_coeff,den_coeff)
w=p.logspace(0,8,801)
ss=1j*w
hf=lambdify(s,V[3],"numpy")
v=hf(ss)
p.loglog(w,abs(v),lw=2)
p.grid(True)
p.show()

#Prob1
A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1/s)
num_coeff,den_coeff=get_coeff(V[3])
H=sp.lti(num_coeff,den_coeff)

t0=np.linspace(0,0.001,10001)
t,y=sp.impulse(H,None,t0)
plt.plot(t,y)
plt.show()
#Prob2
def Vi(t):
    return(np.sin(2e3*pi*t)+np.cos(2e6*pi*t))

A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,1)
num_coeff,den_coeff=get_coeff(V[3])
H=sp.lti(num_coeff,den_coeff)

t0=np.linspace(0,0.01,10001)
t,y,svec=sp.lsim(H,Vi(t0),t0)
plt.plot(t,y)
plt.show()
#Prob3 Function defined already
#Prob4
def V_decay(w,a,t):
    return(np.cos(w*t)*np.exp(-a*t))

A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1)
num_coeff,den_coeff=get_coeff(V[3])
H=sp.lti(num_coeff,den_coeff)

t0=np.linspace(0,0.0001,10001)
t,y,svec=sp.lsim(H,V_decay(1e6,10000,t0),t0)
plt.plot(t,y)
plt.show()
#Prob5
A,b,V=highpass(10000,10000,1e-9,1e-9,1.586,1/s)
num_coeff,den_coeff=get_coeff(V[3])
H=sp.lti(num_coeff,den_coeff)

t0=np.linspace(0,0.001,10001)
t,y=sp.impulse(H,None,t0)
plt.plot(t,y)
plt.show()


#Problem1
A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,V_step)
num_coeff,den_coeff=get_coeff(V[3])
print(num_coeff,den_coeff)
H=sp.lti(num_coeff,den_coeff)

w1,S1,phi=H.bode()
plt.semilogx(w1,S1)
plt.title("Step response of Lowpass filter")
plt.xlabel("w in log scale")
plt.ylabel("|Y(jw)|")
plt.grid()
plt.show()

t0=np.linspace(0,0.001,1000)
t,x=sp.impulse(H,None,t0)
plt.plot(t,x)
plt.title("Time domain step response of Lowpass filter")
plt.xlabel("t")
plt.ylabel("y[t]")
plt.show()

#Problem2
A,b,V=lowpass(10000,10000,1e-9,1e-9,1.586,V_i_s)
V0=V[3]
num_coeff,den_coeff=get_coeff(V[3])
print(num_coeff,den_coeff)
H=sp.lti(num_coeff,den_coeff)

w1,S1,phi=H.bode()
plt.semilogx(w1,S1)
plt.title("Frequency response of Lowpass filter to Vi(t)")
plt.xlabel("w in log scale")
plt.ylabel("|Y(jw)|")
plt.grid()
plt.show()

t0=np.linspace(0.,1.,1000001)
t,x=sp.impulse(H,None,t0)
print(t)
print(x)
print(max(t))
print(max(x))
plt.plot(t,x)
plt.title("Time domain response of Lowpass filter")
plt.xlabel("t")
plt.ylabel("y[t]")
plt.show()

#Problem3
def highpass(R1,R3,C1,C2,G,Vi):
    A=Matrix([[0,G,-G,-1],[0,-G/((G-1)),0,1/((G-1))],[C1*s,(1/R3)-(C2*s),0,0],[s*(C1+C2)+(1/R1),0,(-C2*s),-1]])
    b=Matrix([0,0,0,Vi*G])
    V=A.inv()*b
    return(A,b,V)


#Problem4
s=symbols('s')
A,b,V=highpass(1e4,1e4,1e-9,1e-9,1.586,V_decay_s)
V0=V[3]
#print(V0)
#print(V0)
num_coeff,den_coeff=get_coeff(V[3])
print(num_coeff,den_coeff)
H=sp.lti(num_coeff,den_coeff)

w1,S1,phi=H.bode()
plt.semilogx(w1,S1)
plt.title("Response of Highpass filter to decaying sinusoid")
plt.xlabel("w in log scale")
plt.ylabel("|Y(jw)|")
plt.grid()
plt.show()

t0=np.linspace(0,0.0001,1000)
t,x=sp.impulse(H,None,t0)
plt.plot(t,x)
plt.title("Time domain response of Highpass filter to decaying sinusoid")
plt.xlabel("t")
plt.ylabel("y[t]")
plt.show()


#Problem5
s=symbols('s')
A,b,V=highpass(1e4,1e4,1e-9,1e-9,1.586,V_step)
V0=V[3]
num_coeff,den_coeff=get_coeff(V[3])
print(num_coeff,den_coeff)
H=sp.lti(num_coeff,den_coeff)

w1,S1,phi=H.bode()
plt.semilogx(w1,S1)
plt.title("Step response of Highpass filter")
plt.xlabel("w in log scale")
plt.ylabel("|Y(jw)|")
plt.grid()
plt.show()

t0=np.linspace(0,0.0001,1000)
t,x=sp.impulse(H,None,t0)
plt.plot(t,x)
plt.title("Time domain step response of Highpass filter")
plt.xlabel("t")
plt.ylabel("y[t]")
plt.show()

num_coeff,den_coeff=get_coeff((1/s**2))
print(num_coeff,den_coeff)
num_coeff,den_coeff=get_coeff(V_i_s)
print(num_coeff,den_coeff)
num_coeff,den_coeff=get_coeff(V_step)
print(num_coeff,den_coeff)