print(round( 2*7/9.81,3) )

# Hallemos la posición para V=7 y t=1.1

print(7*1.1-0.5*9.81*(1.1)**2) 

# tiempo al que se da la altura máxima

print(round( 7/9.81,3) )  

7*0.714-0.5*9.81*(0.714)**2 
def ypos(v,t):

    g=9.81

    ls=2*v/g

    if t > 0 and t <= ls:

        y=v*t-0.5*g*t**2

    else:

        y="tiempo fuera de rango" 

    return (t,y)  

print(ypos(7,1.1))

print(ypos(7,0.714))

print(ypos(7,1.5)) 



#t,y=ypos(7,1.1)

t,y=ypos(7,0.714)

print('A un tiempo t=%g s, la altura de la bola es %.2f m' % (t,y)) 
# Cambiar el directorio actual 

#os.chdir('/home/mario/Dropbox/cursos/python/haslwanter')

# saber cuál es el directorio actual

#os.path.abspath(os.curdir)
def ypos2(v,t):

    g=9.81

    ls=2*v/g

    if t > 0 and t <= ls:

        y=v*t-0.5*g*t**2

    else:

        y="tiempo fuera de rango" 

    return (t,y)  



if __name__=='__main__':

    v0=7

    t0=0.714



t,y=ypos2(v0,t0)

print('A un tiempo t=%g s, la altura de la bola es %.2f m' % (t,y))

C=21

F=(9/5)*C+32

print(F)

print('Una temperatura de %g grados Celsius equivale a %.2f grados Fahrenheit ' % (C,F))
def caf (C):

    F=(9/5)*C+32

    return(F)

F=caf(21)

print('Una temperatura de %g grados Celsius equivale a %.2f grados Fahrenheit ' % (C,F))
#import modulo as mod

#mod.ypos(7,0.714)
#from modulo import ypos

#ypos(7,0.714)
#from modulo import * 
from math import sqrt, exp, log, sin

sin(0.25)
import numpy as np

print(np.sin(0.25))

a=[0.25,1.2,0.4]

np.sin(a)
## not run

# sin(a)