import math as m
import cmath
Z0 = 50
B = 2*m.pi #2pi/ramda
l = 1/4 #ramda/4

#eveneven
X1ee = complex(0,-(Z0/m.sqrt(2))/m.tan(B*l/2))
X2ee = complex(0,-Z0/m.tan(B*l/2))
Zinee = (X1ee*X2ee)/(X1ee+X2ee)
S11ee = (Zinee-Z0)/(Zinee+Z0)

#evenodd
X1eo = complex(0,-(Z0/m.sqrt(2))/m.tan(B*l/2))
X2eo = complex(0,Z0*m.tan(B*l/2))
Zineo = (X1eo*X2eo)/(X1eo+X2eo)
S11eo = (Zineo-Z0)/(Zineo+Z0)

#oddodd
X1oo = complex(0,(Z0/m.sqrt(2))*m.tan(B*l/2))
X2oo = complex(0,Z0*m.tan(B*l/2))
Zinoo = (X1oo*X2oo)/(X1oo+X2oo)
S11oo = (Zinoo-Z0)/(Zinoo+Z0)

#oddeven
X1oe = complex(0,(Z0/m.sqrt(2))*m.tan(B*l/2))
X2oe = complex(0,-Z0/m.tan(B*l/2))
Zinoe = (X1oe*X2oe)/(X1oe+X2oe)
S11oe = (Zinoe-Z0)/(Zinoe+Z0)

#Scattering Matrix Row 1
S11 = (S11ee+S11eo+S11oo+S11oe)/4
S12 = (S11ee+S11eo-S11oo-S11oe)/4
S13 = (S11ee-S11eo+S11oo-S11oe)/4
S14 = (S11ee-S11eo-S11oo+S11oe)/4

print(f"S11 = {S11}")
print(f"S12 = {S12}")
print(f"S13 = {S13}")
print(f"S14 = {S14}")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
fnorm = np.array(range(5,16))/10
s11 = []
s12 = []
s13 = []
s14 = []
for x in fnorm:
    l = (1/4)*x
    B = 2*m.pi
    
    #eveneven
    X1ee = complex(0,-(Z0/m.sqrt(2))/m.tan(B*l/2))
    X2ee = complex(0,-Z0/m.tan(B*l/2))
    Zinee = (X1ee*X2ee)/(X1ee+X2ee)
    S11ee = (Zinee-Z0)/(Zinee+Z0)

    #evenodd
    X1eo = complex(0,-(Z0/m.sqrt(2))/m.tan(B*l/2))
    X2eo = complex(0,Z0*m.tan(B*l/2))
    Zineo = (X1eo*X2eo)/(X1eo+X2eo)
    S11eo = (Zineo-Z0)/(Zineo+Z0)

    #oddodd
    X1oo = complex(0,(Z0/m.sqrt(2))*m.tan(B*l/2))
    X2oo = complex(0,Z0*m.tan(B*l/2))
    Zinoo = (X1oo*X2oo)/(X1oo+X2oo)
    S11oo = (Zinoo-Z0)/(Zinoo+Z0)

    #oddeven
    X1oe = complex(0,(Z0/m.sqrt(2))*m.tan(B*l/2))
    X2oe = complex(0,-Z0/m.tan(B*l/2))
    Zinoe = (X1oe*X2oe)/(X1oe+X2oe)
    S11oe = (Zinoe-Z0)/(Zinoe+Z0)

    s11.append(0.25*(S11ee+S11eo+S11oo+S11oe))
    s12.append(0.25*(S11ee+S11eo-S11oo-S11oe))
    s13.append(0.25*(S11ee-S11eo+S11oo-S11oe))
    s14.append(0.25*(S11ee-S11eo-S11oo+S11oe))
    
fig, ax = plt.subplots()   
#plot s11    
ax.plot(fnorm, np.abs(s11))
#plot s12
ax.plot(fnorm, np.abs(s12))
#plot s13
ax.plot(fnorm, np.abs(s13))
#plot s14
ax.plot(fnorm, np.abs(s14))
#graph set
ax.set(xlabel='Normalized frequency')
ax.legend([u'S11',u'S12',u'S13',u'S14'],loc=4)
ax.grid()
plt.show()
import math as m
import cmath
Z0 = 50
dB = 3
C = 10**(-dB/20)
Z0e = Z0*m.sqrt((1+C)/(1-C))
Z0o = Z0*m.sqrt((1-C)/(1+C))
beta = 2*m.pi
L = 1/4
Zinooc = complex(0,Z0o*m.tan(beta*L/2)) 
S11ooc = (Zinooc-Z0)/(Zinooc+Z0)

Zineoc = complex(0,-Z0o/m.tan(beta*L/2)) 
S11eoc = (Zineoc-Z0)/(Zineoc+Z0)

Zinoec = complex(0,Z0e*m.tan(beta*L/2)) 
S11oec = (Zinoec-Z0)/(Zinoec+Z0)

Zineec = complex(0,-Z0e/m.tan(beta*L/2)) 
S11eec = (Zineec-Z0)/(Zineec+Z0)

S11c = (S11eec+S11eoc+S11ooc+S11oec)/4
S12c = (S11eec+S11eoc-S11ooc-S11oec)/4
S13c = (S11eec-S11eoc-S11ooc+S11oec)/4
S14c = (S11eec-S11eoc+S11ooc-S11oec)/4

print(f"S11 = {S11c}")
print(f"S12 = {S12c}")
print(f"S13 = {S13c}")
print(f"S14 = {S14c}")


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
fnorm = np.array(range(5,16))/10
s11c = []
s12c = []
s13c = []
s14c = []
for x in fnorm:
    L = (1/4)*x
    beta = 2*m.pi
    Z0e = Z0*m.sqrt((1+C)/(1-C))
    Z0o = Z0*m.sqrt((1-C)/(1+C))

    Zinooc = complex(0,Z0o*m.tan(beta*L/2)) 
    S11ooc = (Zinooc-Z0)/(Zinooc+Z0)

    Zineoc = complex(0,-Z0o/m.tan(beta*L/2)) 
    S11eoc = (Zineoc-Z0)/(Zineoc+Z0)

    Zinoec = complex(0,Z0e*m.tan(beta*L/2)) 
    S11oec = (Zinoec-Z0)/(Zinoec+Z0)
    
    Zineec = complex(0,-Z0e/m.tan(beta*L/2)) 
    S11eec = (Zineec-Z0)/(Zineec+Z0)

    s11c.append(0.25*(S11eec+S11eoc+S11ooc+S11oec))
    s12c.append(0.25*(S11eec+S11eoc-S11ooc-S11oec))
    s13c.append(0.25*(S11eec-S11eoc-S11ooc+S11oec))
    s14c.append(0.25*(S11eec-S11eoc+S11ooc-S11oec))
    
fig, ax = plt.subplots()   
#plot s11    
ax.plot(fnorm, np.abs(s11c))
#plot s12
ax.plot(fnorm, np.abs(s12c))
#plot s13
ax.plot(fnorm, np.abs(s13c))
#plot s14
ax.plot(fnorm, np.abs(s14c))
#graph set
ax.set(xlabel='Normalized frequency')
ax.legend([u'S11',u'S12',u'S13',u'S14'],loc=4)
ax.grid()
plt.show()