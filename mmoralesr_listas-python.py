mylista=[2,"a",3]

print(mylista)

l1=[-5,-4,-3,-2,-1,1,2,3,4,5]

print(l1,type(l1))
# Cuarto elemento de la lista 

l1[3]
# Adicionar un elemento al final de la lista 

l1.append(6)

print(l1)

# insertar elemento en la posición 5 

print("Se inserta 0 en la posición 5")

l1.insert(5,0)

print(l1)
# insertar -6 como primer elemento de la lista 

l1.insert(0,-6)

print(l1)
l2=[7,8,9] 

l3=l1+l2

print(l3)
# borra el elemento con indice 6 en l3 (corresponde al cero)

del l3[6]

l3
print(len(l1))

print( len(l2) )

len(l3)
# borrar el último elemento de una lista 

del l3[len(l3)-1] # ¿porqué len(l3)-1 y no solo len(l3)? 

l3
nom=["Mario","Nelsy","Liliana","Andrea"]

nom.remove("Mario")

print(nom)

l5=[2,4,1,1,1,5,9]

l5.remove(1)

l5
l5=[2,4,1,1,1,0,9]

## Se le entrega un indice 

l6=l5.pop(5)

print("Lo que regresa",l6)

print("Lista actualizada ",l5)

l7=l5.pop()

print("Lo que retorna ",l7)

print("Lista actualizada ",l5)
# recupera el indice para -1. Nota: si hay mas de un -1 regresa el indice del primero 

l3.index(-1) 
-1 in l3 
-1 in l2
print(l3)

print(l3[-1])

print(l3[-2])

print(l3[-len(l3)])
n=21

GradosC=[-5+i*0.5 for i in range(n)]

print(GradosC)

print(range(n))

print(list(range(n)))

print(GradosC)
GradosC=(-5+i*0.5 for i in range(n))

#print(GradosC,type(GradosC))

#GradosC[0]

#list(GradosC)
# Convierte las temperaturas de GradosC que están en grados centígrados a grados Farenheit

GradosF=[9.0/5*C+32 for C in GradosC]

print(GradosF)
GradosC=list(range(-25,41,5)) 

#print(list(GradosC))

GradosF=[9.0/5*C+32 for C in GradosC]

#print(GradosF)

tabla=[GradosC,GradosF]

print(tabla)
tabla[0][2]
print(GradosC)

print(GradosC[5:])

# L[i:j] desde el indice i hasta el indice j-1 

print(GradosC[4:9])

# L[:j] desde el indice 0 (inicio) hasta el indice j-1 

print(GradosC[:9])

# todos menos el primero y el último

GradosC[1:-1]
Gc=GradosC[:]

print(Gc)

print(Gc is GradosC)

Gc2=GradosC

print(Gc is GradosC)

Gc2==GradosC
# Una lista de valores lógicos 

print([i>0 for i in l3])

## lista vacía 

c=[]
nom=["Mario","Nelsy","Liliana","Andrea"]

nom.sort()

print(nom)
t1=("a","e","i","o","u")

print(t1,type(t1))

lt1=list(t1)

print(lt1,type(lt1))
from math import sin

a=[0.25,1.2,0.4]

seno=[sin(i) for i in a ]

print(seno)