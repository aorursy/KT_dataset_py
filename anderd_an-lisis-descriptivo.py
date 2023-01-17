import pandas as pd

data= pd.read_excel('../input/weightchangeduringquarantine/CambioPesoCuarentena.xls')

datac=data.dropna(thresh=7)

datac['PesoInicial'].describe()
datac['PesoFinal'].describe()
(datac

    .groupby(['Sexo'])

    .agg({

       'PesoInicial': ['min','mean','median','max','var','std']})

    )

(data

    .groupby(['Sexo'])

    .agg({

       'PesoFinal': ['min','mean','median','max','var','std']})

    )
(data

    .groupby(['Sexo','ActFisAntes'])

    .agg({

       'PesoInicial': ['min','mean','median','max','var','std']})

    )
(data

    .groupby(['Sexo','ActFisDurante'])

    .agg({

       'PesoFinal': ['min','mean','median','max','var','std']})

    )
(data

    .groupby(['Sexo','Lugar'])

    .agg({

       'PesoInicial': ['min','mean','median','max','var','std']})

    )
(data

    .groupby(['Sexo','Lugar'])

    .agg({

       'PesoFinal': ['min','mean','median','max','var','std']})

    )
import numpy as np

minimo=datac['PesoInicial'].min()

"minimo", datac['PesoInicial'].min()

maximo=datac['PesoInicial'].max()

"maximo", datac['PesoInicial'].max()

Ancho=(maximo-minimo)/7 

print(Ancho)

print(pd.cut(datac['PesoInicial'],bins=7, include_lowest=True).value_counts())



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import scipy.stats as stats

import seaborn as sns

b=np.arange(39.6,100,7.914285714285714)

n, bins, patches=plt.hist(datac['PesoInicial'], bins=b)

mids=[(bins[i+1]+bins[i])/2 for i in np.arange(len(bins)-1)]

### habrá forma de obtenerlos directamente con lo que entrega plt.hist

for i in np.arange(len(mids)):

    plt.text(mids[i]-1,n[i]+0.5,round(n[i]))
import numpy as np

minimo=datac['PesoFinal'].min()

"minimo", datac['PesoFinal'].min()

maximo=datac['PesoFinal'].max()

"maximo", datac['PesoFinal'].max()

Ancho=(maximo-minimo)/7 

print(Ancho)

print(pd.cut(data['PesoFinal'],bins=7, include_lowest=True).value_counts())



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import scipy.stats as stats

import seaborn as sns

b=np.arange(38.8,105,8.742857142857144)

n, bins, patches=plt.hist(datac['PesoFinal'], bins=b)

mids=[(bins[i+1]+bins[i])/2 for i in np.arange(len(bins)-1)]

### habrá forma de obtenerlos directamente con lo que entrega plt.hist

for i in np.arange(len(mids)):

    plt.text(mids[i]-1,n[i]+0.5,round(n[i]))
b1=np.arange(39.6,100,7.914285714285714)

n, bins, patches=plt.hist(datac['PesoInicial'].fillna(0), bins=b1,label='PesoInicial',histtype='step')



b2=np.arange(38.8,105,8.742857142857144)

n, bins, patches=plt.hist(datac['PesoFinal'].fillna(0), bins=b2,label='PesoFinal',histtype='step')

    

plt.legend(loc='upper right')

plt.show() 
print(np.percentile(datac['PesoInicial'], [25, 50, 75]))



print(np.percentile(datac['PesoFinal'], [25, 50, 75]))



plt.boxplot([datac['PesoInicial'],datac['PesoFinal']], sym='*',labels=["PesoInicial", "PesoFinal"])

plt.ylabel("PESOS")

plt.show()