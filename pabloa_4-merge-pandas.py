import pandas as pd
Nombres = pd.DataFrame({'id':[1,2,3,4], 'Nombre': ["Ana", "Juan", "Carolina", "Pedro"]})

Nombres
Nombres2 = pd.DataFrame({'id':[5,6], 'Nombre': ["Julia", "Alberto"]})

Nombres2
Nombres.append(Nombres2)
Nombres = pd.DataFrame({'id':[1,2,3,4], 'Nombre': ["Ana", "Juan", "Carolina", "Pedro"]})

Nombres
Edad = pd.DataFrame({'id':[1,2,3,4], 'Edad':[11,21,8,15]})

Edad
pd.merge(Nombres, Edad)
Nombres = pd.DataFrame({'id_Nombres':[1,2,3,4], 'Nombre': ["Ana", "Juan", "Carolina", "Pedro"]})

Edad = pd.DataFrame({'id_edades':[1,2,3,4], 'Edad':[11,21,8,15]})
pd.merge(Nombres, Edad, left_on='id_Nombres', right_on='id_edades')
Nombres = pd.DataFrame({'id':[3,4,5,6], 'Nombre': ["Ana", "Juan", "Carolina", "Pedro"]})

Edad = pd.DataFrame({'id':[1,2,3,4], 'Edad':[11,21,8,15]})
pd.merge(Nombres, Edad, how='inner')
pd.merge(Nombres, Edad, how='outer')
pd.merge(Nombres, Edad, how='left')
pd.merge(Nombres, Edad, how='right')