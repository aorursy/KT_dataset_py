# Se importan los paquetes estandar 

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import scipy.stats as stats

import seaborn as sns



# Generación de datos

# 500 datos de una distribución normal estandar 

# si quieres uniforme quita la n 

x = np.random.randn(500) 

# Comando plot 

plt.plot(x,'.')

# Se muestra el gráfico

plt.show()
# otra forma 

plt.scatter(np.arange(len(x)),x)

plt.show()
plt.hist(x, bins=25)

plt.show()
# Ingreso de datos 

t1_1=[105,221,183,186,121,181,180,143,

97,154,153,174,120,168,167,141,

245,228,174,199,181,158,176,110,

163,131,154,115,160,208,158,133,

207,180,190,193,194,133,156,123,

134,178,76,167,184,135,229,146,

218,157,101,171,165,172,158,169,

199,151,142,163,145,171,148,158,

160,175,149,87,160,237,150,135,

196,201,200,176,150,170,118,149]
plt.plot(np.array(t1_1))

plt.show()
# con los límites de intervalos de clasen desde 70 a 250 de 20 en 20 

# histograma de frecuencias absolutas 

b=np.arange(70,270,20)

print(b)

plt.hist(t1_1,bins=b)

plt.show()
# con los límites de intervalos de clase desde 70 a 250 de 20 en 20 

# histograma de frecuencias relativas/long de clase  

plt.hist(t1_1, bins=b, density=True) 

plt.show()
# con los límites de intervalos de clasen desde 70 a 250 de 20 en 20 

# histograma de frecuencias absolutas acumuladas  

plt.hist(t1_1, bins=b,cumulative=True)

plt.show()
## histtype

plt.hist(t1_1, bins=b, histtype='barstacked')

plt.show() 
##  histtype='step'

plt.hist(t1_1, bins=b, histtype='step')

plt.show()  
##  histtype='stepfilled'

plt.hist(t1_1, bins=b, histtype='stepfilled',facecolor='g', alpha=0.75)

#plt.text()

plt.show()  
n, bins, patches=plt.hist(t1_1, bins=b)

print(n)

#print(bins,type(bins))

#print(dir(patches))

### obtener las marcas de clase



### forma poco eficiente pero 

### funciona



#s=np.diff(bins)/2

#print(s)

#s=np.append(s,[0])

#print(s)

#mids=bins+s

#print(mids)

#mids=mids[:(len(mids)-1)]

#print(mids)

#plt.ylim([0,25])



#### una forma más eficiente 

mids=[(bins[i+1]+bins[i])/2 for i in np.arange(len(bins)-1)]

### habrá forma de obtenerlos directamente con lo que entrega plt.hist

for i in np.arange(len(mids)):

    plt.text(mids[i]-1,n[i]+0.5,round(n[i]))

# habrá alguna forma de poner las frecuencias directamente

# con lo que entrega plt.hist?

plt.ylim([0,25])

plt.show()  
#dir(patches)
sns.kdeplot(t1_1);
fig, axs = plt.subplots(1,1)

sns.distplot(t1_1, rug=True)

plt.show()
sns.distplot(t1_1, rug=True,hist=False);
from scipy.stats import norm

sns.distplot(t1_1,fit=norm, rug=True,hist=True);
ax = sns.distplot(x, fit=norm, rug=True, rug_kws={"color": "g"},

                  fit_kws={"color": "r", "lw": 2, "label": "NORM"},

                  kde_kws={"color": "k", "lw": 3, "label": "KDE"},

                  hist_kws={"histtype": "step", "linewidth": 3,

                            "alpha": 0.7, "color": "g"})
fig, axs = plt.subplots(1,2)

sns.distplot(t1_1, rug=True,ax=axs[0])

axs[1].hist(t1_1, bins=b,density=True)

#n=len(t1_1)

#print(n)

#s=np.array(t1_1).std(ddof=1)

#h = ((4*s**5)/(3*n))**(1/5) 

#h=1.06*s*n**(-1/5)

#print(h)

kde_small = stats.kde.gaussian_kde(t1_1, 0.1)

kde = stats.kde.gaussian_kde(t1_1)

kde_large = stats.kde.gaussian_kde(t1_1, 1)

t1_1.sort()

axs[1].plot(t1_1, kde.evaluate(t1_1), 'r')

axs[1].plot(t1_1,kde_small.evaluate(t1_1),'-', color=[0.8, 0.8, 0.8])

axs[1].plot(t1_1,kde_large.evaluate(t1_1),'--')

plt.show() 
plt.plot(stats.cumfreq(t1_1,9)[0])

plt.show()
plt.plot(mids, stats.cumfreq(t1_1,9)[0])

plt.show()
plt.hist(t1_1, bins=b,cumulative=True)

plt.plot(mids, stats.cumfreq(t1_1,9)[0],lw=3)

plt.show()


# simular datos 

import random as rnd 

x = np.random.randn(250) # control

y = np.random.randn(250)+3 # tratados 

b=np.repeat("Trat",250)

a=np.repeat("Cont",250)

grupos=np.concatenate((a,b))

d=np.concatenate((x,y))

datosh=pd.DataFrame({'d':d,'Trat':grupos})

datosh.head()

# si quieres los histogramas separados 

#datosh.hist(by='Trat');

datosh[datosh.Trat=='Trat'].d

plt.hist(datosh[datosh.Trat=='Trat'].d, bins=10,histtype='step', alpha=0.5, label='Trat', color='g')

plt.hist(datosh[datosh.Trat=='Cont'].d, bins=10,histtype='step', alpha=0.5, label='Cont', color='r')

plt.legend(loc='upper right')

plt.show()
# Ingreso de datos del experimento 1

exp1=np.array([16.85,16.40,17.21,16.35,16.52,17.04,16.96,17.15,16.59,16.57]) 

# Ingreso de datos del experimento 2

exp2=np.array( [17.50,17.63,18.25,18.00,17.86,17.75,18.22,17.90,17.96,18.15])



x=[1,2]

y=[exp1.mean(),exp2.mean()]

n=[len(exp1), len(exp2)]

error=[1.96*exp1.std(ddof=1),1.96*exp2.std(ddof=1)]/np.sqrt(n)
plt.errorbar(x,y, yerr=error, fmt='o',capsize=5, capthick=3)

plt.show()
d=np.concatenate((exp1 , exp2))

#print(d)

exp=np.repeat(["Exp1","Exp2"], [10,10])

datos=pd.DataFrame({'res':d,'Exp':exp})

datos
grp=datos.groupby('Exp')

medias=grp.mean()

medias

desv=grp.std(ddof=1)

n=grp.count() 

error=1.96*desv/np.sqrt(n) 

error

#medias

#desv

#n
plt.errorbar(["Exp1","Exp2"],medias.res, yerr=error.res, capsize=5, capthick=3)

plt.show()
plt.errorbar(["Exp1","Exp2"],medias.res, yerr=error.res,fmt='o', capsize=5, capthick=3)

plt.show()
#Q_3+1.5(Q_1-Q_3) límites para los bigotes 

#Q_1-1.5(Q_1-Q_3)

plt.boxplot(t1_1,sym='*')

plt.ylabel("Resitencia")

plt.show()
plt.boxplot(t1_1,vert=False, sym='*')

plt.xlabel("Resistencia")

plt.show()
plt.boxplot([exp1,exp2], sym='*')

plt.xlabel("Experimento")

plt.ylabel("Resistencia")

plt.show()
sns.violinplot(t1_1);
sns.violinplot(y=t1_1)

plt.show()
#print(datos.head())

sns.violinplot(x="Exp",y="res",data=datos)

plt.xlabel("Experimento")

plt.ylabel("Resistencia")

plt.show()