mytupla0=('abc',4,True,5,False,2.5)

mytupla=(1,0.7,mytupla0,"Hola")

print(mytupla0)

mytupla
mytupla[0] # primer elemento de la tupla. Note que inicia en cero 
mytupla[2] # tercer  elemento 
mytupla[3] # cuarto elemento  
# Cuidado!!! 

#mytupla[4] # IndexError: tuple index out of range

# no corre, a eso se refiere con inmutable 

# mytupla[2]=0 #TypeError: 'tuple' object does not support item assignment

# mytupla[2]='xyz'
type(mytupla) 
point = 10, 20

print(point,type(point))
x,y=point

print("x= ",x)

print("y= ",y)
import scipy.stats as s 

n = s.norm(0,10) # crea una normal de media 0 y desviación estandar 10

res=s.shapiro(n.rvs(100)) # el segundo es  el pvalor 

print(res,type(res))
W,p=res # el primer elemento en W el segundo en p 
print("W= ",W,"P-valor= ",p)

# recuerda la otra frma de formatear la respuestra 

print("El estadístico de Shapiro - Wilks es %0.3f, con p_valor igual a %0.4f" %res)
a=(3,2,4,1,6,4) # tupla original 

b=a[1:4] # tupla creada a partir de a 

print(b,type(b)) 
a[:4] # los tres primeros, índices 0, 1 y 2 
a[2:] # desde el tercero (recuerde que inicia en cero) hasta el final 
a[::-1] # reversa la tupla 

b=(1,3) 
len(a)  # Número de elementos de la tupla 

print(len(mytupla))

print(mytupla)

a[len(a)-1] ## último elemento de la tupla a 
a.index(4)  # ¿en qué posición de la tupla está el primer valor 4? 
a.count(4) # ¿cuántas veces está 4?
print(4 in a) # sí está (True)

print("x" in a ) # no está (False) 
tupla1=(10,13,15,12)

tupla2=("Jose","María","Pedro","Juan")

tupla3=(31,tupla1,14,tupla2,20)

print(tupla3) 
tuplaS=tupla1+tupla2

print(tuplaS)
Tupla3=(2,"xyz",[3,5,7],True)

print(Tupla3[2]) # recupera el tercer elemento de la tupla 

Tupla3[2].append(10)  

# ¿Se ha modificado la tupla? no que no se podía  

Tupla3 