import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.cluster import KMeans
df = pd.DataFrame({

    'Nome':["Oliver","Harry","Jack","George","Noah","Charlie","Jacob","Alfie","Freddie","Oscar"],

    'Idade':[20,22,23,27,28,31,33,35,37,38],

    'Salario':[1000.00,1200.00,3000.00,7000.00,1800.00,1100.00,3300.00,5000.00,5700.00,3800.00]

})

df.describe()
df.plot.scatter(x='Idade', y='Salario')
df.plot.hist(y='Idade')
kmeans = KMeans(n_clusters=3)

kmeans = kmeans.fit(df.Idade.values.reshape(-1, 1))

labels = kmeans.predict(df.Idade.values.reshape(-1, 1))



C = kmeans.cluster_centers_

print(labels, C)
dfIdadeGrupo = pd.concat([df,pd.DataFrame(labels, columns = ['Grupo'])], axis=1, join='inner')

dfIdadeGrupo
cores = dfIdadeGrupo.Grupo.map({0:'b',1:'r',2:'y'})

dfIdadeGrupo.plot.scatter(x='Idade',y='Salario', c=cores)
kmeans = KMeans(n_clusters=3)

kmeans = kmeans.fit(df.Salario.values.reshape(-1, 1))

labels = kmeans.predict(df.Salario.values.reshape(-1, 1))



C = kmeans.cluster_centers_

print(labels, C)
dfSalarioGrupo = pd.concat([df,pd.DataFrame(labels, columns = ['Grupo'])], axis=1, join='inner')

dfSalarioGrupo.sort_values(by='Salario')
cores = dfSalarioGrupo.Grupo.map({0:'b',1:'r',2:'y'})

dfSalarioGrupo.plot.scatter(x='Salario',y='Idade', c=cores)
kmeans = KMeans(n_clusters=3)

# kmeans = kmeans.fit(df[['Idade','Salario']].values.reshape(-1, 1))

# labels = kmeans.predict(df[['Idade','Salario']].values.reshape(-1, 1))

kmeans = kmeans.fit(df[['Idade','Salario']].values)

labels = kmeans.predict(df[['Idade','Salario']].values)

C = kmeans.cluster_centers_

print(labels,C)


dfGrupo = pd.concat([df,pd.DataFrame(labels, columns= ['Grupo'])], axis=1, join='inner')

dfGrupo.sort_values(by='Grupo')
cores = dfGrupo.Grupo.map({0:'b',1:'r',2:'y'})

dfGrupo.plot.scatter(x='Idade',y='Salario', c=cores)