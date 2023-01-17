import pandas as pd

import numpy as np

import matplotlib

import matplotlib.pyplot as plt

import random as rnd 
exp=rnd.choices(['E', 'F'], weights=[0.5,0.5])

#print(exp)

n=10

p=0.75 

for i in np.arange(n):

    exp=rnd.choices(['E', 'F'],weights=[p,1-p])

    print(i," ",exp)



# simular una bernoulli    

    

def bernoulli(p,k=1):

    if(p<0 or p>1):

        return("La probabilidad debe ser un número entre 0 y 1")

    else: 

        return(rnd.choices([1, 0],k=k,weights=[p,1-p]))



bernoulli(0.3,k=10)

# una vez

exp=rnd.choices(['C', 'S'], weights=[0.5,0.5])

#print(exp)



# n monedas  

n=3 

p=0.75 # probabillidad de sello 

res=[] # para guardar los resultados 

for i in np.arange(n):

    res+=rnd.choices(['C', 'S'],weights=[p,1-p])



print(res)

# contar el número de sellos 

res=np.array(res)

sum(res=='S')



# ahora repetiremos el código anterior R veces 

R=8000 # numero de repeticiones (las veces que se lanzan las n monedas)

n=3 # número de monedas  

#p=0.5 # probabillidad de sello 



NumSellos=[] # para guardar los valores de la variable # de sellos 



for j in np.arange(R):

    res=[] # para guardar los resultados 

    for i in np.arange(n):

        res+=rnd.choices(['S', 'C'],weights=[p,1-p])

    res=np.array(res)

    NumSellos.append(sum(res=='S'))



X=np.array(NumSellos)

df=pd.DataFrame({'X':X})

# frec observadas 

obs=df['X'].value_counts(sort=False)/R

obs
import itertools 

m=['S', 'C']

### todos los posibles resultados 

S0=[f'{a}{b}{c}' for a, b, c in itertools.product(m,m,m)]

### se organizan en un dataframe 

S=pd.DataFrame({'Ei':S0})

S

# numero de elementos en S 

n=S.shape[0] 

### cuantos 'S' hay en cada fila 

#S.Ei.str.count('S')

S['X']=S['Ei'].str.count('S')

#print(S.head())

#print(S)

DistProb=S.groupby('X').count()/n

DistProb['Oi']=obs

DistProb
DistProb.plot(kind='bar',alpha=0.8,rot=0)

#for i in np.arange(len(DistProb.index)):

#    plt.text(DistProb.index[i]-0.5,DistProb.Ei[i]+0.5,round(DistProb.Ei[i],3))

plt.legend(loc=0)

plt.xlabel("Número de sellos")

plt.ylabel("Probabilidad") 

plt.show()
#plt.bar(DistProb.index,DistProb.Ei)

#plt.bar(DistProb.index+0.5,DistProb.Oi)

#for i in np.arange(len(DistProb.index)):

#    plt.text(DistProb.index[i]-0.5,DistProb.Ei[i]+0.5,round(DistProb.Ei[i],3))
width = 0.35

x = np.arange(DistProb.shape[0])

fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, DistProb['Ei'], width, label='Esperado')

rects2 = ax.bar(x + width/2, DistProb['Oi'], width, label='Observado')

ax.set_ylabel('Probabilidades')

ax.set_title('Probs esperadas y observadas')

ax.set_xticks(x)

ax.set_xticklabels(x)

ax.legend()

plt.show()

# para colocar encima las probabilidades encima de cada barra 

# ****no funciona **** 

# si lo pone a funcionar y me explica que pasa tiene bono

def autolabel(rects):

    """Attach a text label above each bar in *rects*, displaying its height."""

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(0, 3),  # 3 points vertical offset

                    textcoords="offset points",

                    ha='center', va='bottom')



autolabel(rects1)

autolabel(rects2)

fig.tight_layout()

plt.show()
p=0.3 # probabillidad de Exito  

res=['Fr'] # para guardar los resultados 

i=0

imp=False ## si quieres ver los resultados pon True aquí 

while 'Fr' in res:

  res=rnd.choices(["Ex", "Fr"],weights=[p,1-p])

  i += 1

  if imp:

    print(res,i)
### simulación

R=10000 # numero de repeticiones del experimento   

X=[]

for j in np.arange(R):

    res=['Fr'] # para guardar los resultados 

    i=0

    imp=False ## si quieres ver los resultados pon True aquí 

    while 'Fr' in res:

      res=rnd.choices(["Ex", "Fr"],weights=[p,1-p])

      i += 1

      if imp:

        print(res,i)

    X.append(i)
df=pd.DataFrame({'X':X})

pr_obs=df["X"].value_counts()/R

pr_obs.head()

#df.head()

xo=pr_obs.index

dfo=pd.DataFrame({'x':xo,'pr':pr_obs})

dfo1=dfo.sort_values(by=['x'])

dfo1.head()

dfo1.tail()
### Lo teórico 

# para calcular la función de probabilidad 

# recuerda que el rango de esta variable es

# el conjunto de los naturales por eso se hace hasta la media mas 

# 3 desciaviones estándar 

#p=0.01 # probabilidad de éxito 

Ex=1/p # valor esperado teórico 

Vx=(1-p)/p**2 # varianza teórica 

dfo1["prt"]=(1-p)**(dfo1.x-1)*p #probabilidades teóricas 

dfo1.head()
### para no usar todos los valores de x que aparecen

### en la simulación. Tomaré los menores que la media más

### dos desviaciones estándar 

lim=Ex+2*np.sqrt(Vx)

dfo1=dfo1.iloc[range(0,round(lim)),]

dfo1[["pr","prt"]].plot(kind="bar",alpha=0.8,rot=90)

#for i in np.arange(len(DistProb.index)):

#    plt.text(DistProb.index[i]-0.5,DistProb.Ei[i]+0.5,round(DistProb.Ei[i],3))

xtpaso=20

plt.xticks(list(range(1,max(dfo1.x)+1,xtpaso)),[str(i) for i in range(1,max(dfo1.x)+1,xtpaso)])

plt.legend(loc=0)

plt.xlabel("Número de ensayos")

plt.ylabel("Probabilidad") 

plt.show()

d={(i,j):i+j for i in range(1,7) for j in range(1,7)}

df=pd.DataFrame.from_dict(d,orient='index')

df.columns=['Suma']

df.head()