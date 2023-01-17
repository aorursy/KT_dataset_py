import pandas as pd

import random as rnd 

### se genera una columna con valores de sexo de manera 

### aleatoria para ejemplos ilustrativos

genero=rnd.choices(['f', 'm'], [100,100], k=169) 

### Lectura de los datos de Bioestadística de Daniel 

datos=pd.DataFrame( {'Edad':[27,48,27,63,53,43,24,26,56,39,24,26,40

,27,32,63,30,33,34,45,27,31,32,51,30,28

,42,35,44,21,24,28,26,30,26,25,30,27,29

,23,22,22,45,25,23,29,32,23,32,41,26,29

,37,23,44,28,43,61,48,43,18,23,47,36,24

,47,37,45,42,39,24,34,29,38,47,30,24,28

,30,33,40,40,40,29,41,24,53,34,42,50,22

,27,26,48,26,22,32,53,50,40,26,33,31,50

,47,29,36,29,25,38,30,30,23,46,31,42,30

,41,26,51,48,21,62,27,31,24,21,29,34,38

,19,52,31,53,26,25,22,30,18,19,37,27,28

,52,20,28,27,22,34,27,24,49,37,40,28,23

,48,37,44,38,48,46,38,26,49,36,31,31,39], 'Genero':genero} ) 

datos.head()
print("minimo", datos.Edad.min())

print("maximo", datos.Edad.max())

# Categorizar la variable edad 

datos['GrEdad'] = pd.cut(datos['Edad'], bins=[17, 26, 59, float('Inf')], labels=['Joven', 'Adulto', 'Viejo'])

#datos[datos.Edad==18]
datos.head()
datos.GrEdad.value_counts() 
### Se agrupa por género y grupo de edad. 

datosgrp=datos.groupby(['Genero','GrEdad']) 

datosgrp.count()
## media de edad para cada combinación de genero y grupo de edad

datosgrp.mean()
datosgrp.median()
datosgrp.var()
datosgrp.std()
datosgrp.quantile([0,0.25,0.5,0.75,1])
datosgrp.describe()