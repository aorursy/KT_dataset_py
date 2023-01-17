from sympy import *

import sympy

from sympy.plotting import plot
x=Symbol('x')

funcion="x**4+x**2+x+1 "

derivada1=diff(funcion,x)

derivada2=diff(funcion,x,2)

print("Funcion: ",funcion)

print("Primera derivada: ",derivada1)

print("Segunda derivada: ",derivada2)


iteraccion=0

error=100

numeroAnt=0

numeroAct=0

while error>1 and iteraccion<=10:

    ndev1=sympy.sympify(derivada1).subs(x,numeroAct);

    ndev2=sympy.sympify(derivada2).subs(x,numeroAct);

    numeroAnt=numeroAct   

    numeroAct=(numeroAct-(ndev1/ndev2))

    if numeroAnt==0:

        error=100        

    else:

        error=abs((numeroAnt-numeroAct)/numeroAnt)*100        

    print("Iteracion: ",iteraccion," Raiz: ",round(numeroAct,5), " El error es de: ",round(error,5))

    iteraccion+=1    

    



#ejey=funcion.evalf(subs={x:numeroAct})

ejey=sympy.sympify(funcion).subs(x,numeroAct);

minOmax=(derivada2.evalf(subs={x:numeroAct}))

print('Las coordenadas del punto son: (',numeroAct,' , ',ejey,')')

if minOmax>=0:

    print("Es un valor  minimo")

else:

    print ('Es un valor maximo')

print('Las coordenadas del punto son: (',numeroAct,' , ',ejey,')')

grafica=plot(funcion,"x" ,autoscale=True )

print("Punto mas general")

#Metodo que nos retorna evaluacion de un numero en la función

def graf (v):

    return sympy.sympify(funcion).subs(x,v);

        

a=[]

b=[]



for i in range(round(numeroAct)-3,round(numeroAct)+3):

    a.append(i)

    b.append(graf(i))

from matplotlib import pyplot



pyplot.plot(a,b, color='black');

pyplot.plot(numeroAct,ejey,'o', color='blue');

pyplot.axhline(0, color="black");

pyplot.axvline(0, color="black");


