#importaciones
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
models = pd.read_csv('../input/events.csv',low_memory = False)
models.head()
models.columns
#10 modelos mas vistos
models['model'] = models['model'].dropna()
modelos = models[(models['event'] == 'viewed product') & (models['model'])].groupby('model')['event'].count()
modelos1 = pd.DataFrame(data = modelos)
modelos2 = modelos1.reset_index().sort_values(by= 'event', ascending = False)
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
ax = sns.catplot(x="model", y="event", kind = "bar", palette=flatui, data = modelos2.head(10))
ax.set_xticklabels(modelos2.head(10)['model'].values,rotation=90)

plt.title('10 modelos con mas vistas', fontsize = 18)
plt.ylabel('Cantidades', fontsize = 16)
plt.xlabel('Modelos', fontsize = 16)

convXModelo = models[(models['event'] == 'conversion') & (models['model'])].groupby('model')['event'].count()
convXModelo1 = pd.DataFrame(data = convXModelo)

convXModelo2 = convXModelo1.reset_index().sort_values(by= 'event', ascending = False)
convXModelo2.rename(columns = {'event' : 'cantidadConversiones'},inplace = True)
convXModelo2

#111 modelos con conversiones
ax = sns.catplot(x="model", y="cantidadConversiones", palette="GnBu_d", kind ='bar', data = convXModelo2.head(10))
ax.set_xticklabels(convXModelo2.head(10)['model'].values,rotation=90)

plt.title('10 dispositivos con mas conversiones', fontsize = 18)
plt.ylabel('Cantidad', fontsize = 16)
plt.xlabel('Modelos', fontsize = 16)

#10 dispositivos con mas checkouts
dmc = models[(models['event'] == 'checkout') & (models['model'])].groupby('model')['event'].count()
dmc = pd.DataFrame(data = dmc.sort_values(ascending =False).head(10)).reset_index()
#creo el dataframe de los 10 dispositivos con mas checkouts
checksPorModel = models[(models['model'].isin(np.array(dmc['model'])))]
#Grafico de los 10 dispositivos con mas checkouts y la cantidad en eventos
ax = sns.catplot(y="model", hue="event", kind="count", palette="pastel", edgecolor=".6",\
                data=checksPorModel, height = 12,aspect = 1, ci = 95);
#Cambio la proporcion (viewed products es muy alto ~ 400000) 
plt.xscale('log')
#Seteo labels
plt.title('Grafico de los 10 dispositivos con mas checkouts y la cantidad en eventos', fontsize = 18)
plt.ylabel('Modelos', fontsize = 16)
plt.xlabel('Cantidad (log)', fontsize = 16)
#10 dispositivos con mas leads
dml = models[(models['event'] == 'lead') & (models['model'])].groupby('model')['event'].count()
dml = pd.DataFrame(data = dml.sort_values(ascending =False).head(10)).reset_index()
#creo el dataframe de los 10 dispositivos con mas leads
leadsForModel = models[(models['model'].isin(np.array(dml['model'])))]
#Grafico de los 10 dispositivos con mas leads y la cantidad en eventos
ax = sns.catplot(y="model", hue="event", kind="count", palette="hls", edgecolor=".6",\
                data=leadsForModel, height = 12,aspect = 1, ci = 95);
#Cambio la proporcion (viewed products es muy alto) 
plt.xscale('log')
#Seteo labels
plt.title('Grafico de los 10 dispositivos con mas leads y la cantidad en eventos', fontsize = 18)
plt.ylabel('Modelos', fontsize = 16)
plt.xlabel('Cantidad (log)', fontsize = 16)
#Cantidad de modelos por condicion sin nulos
models['condition'].dropna().value_counts()
modelosEnMejoresCond = models[(models['condition'] == 'Excelente') & (models['model'])]
modelosEnMejoresCond = pd.crosstab(modelosEnMejoresCond.condition,modelosEnMejoresCond.model).max().sort_values(ascending = False)
modelosEnMejoresCond
## 10 modelos en 'peores' condiciones (Bom)
models['condition'].value_counts()
modelosEnPeoresCond = models[(models['condition'] == 'Bom - Sem Touch ID') & (models['model'])]
modelosEnPeoresCond = pd.crosstab(modelosEnPeoresCond.condition,modelosEnPeoresCond.model).max().sort_values(ascending = False)
modelosEnPeoresCond
modelosMasNuevos = models[(models['condition'] == 'Novo') & (models['model'])]
modelosMasNuevos = pd.crosstab(modelosMasNuevos.condition,modelosMasNuevos.model).max().sort_values(ascending = False)
modelosMasNuevos

ConditionsAndCheckoutsModels = models[(models['event'] =='checkout')]
arrayLabels = dmc['model'].values
ConditionsAndCheckoutsModels = ConditionsAndCheckoutsModels[(ConditionsAndCheckoutsModels['model'].isin(arrayLabels))]
ConditionsAndCheckoutsModels = pd.crosstab(ConditionsAndCheckoutsModels.condition, \
                                   ConditionsAndCheckoutsModels.model)
#plot
fig,ax = plt.subplots(figsize=(10,7))
g = sns.heatmap(ConditionsAndCheckoutsModels,  cmap="Blues", ax=ax,cbar_kws={'label': 'Cantidades'})

plt.title('10  modelos con mas checkouts y mayor cantidad por condicion', fontsize = 18)
plt.ylabel('Conditions', fontsize = 18)
plt.xlabel('Models', fontsize = 18)

ConditionsAndConversionsModels = models[(models['event'] =='conversion')]
arrayLabels = convXModelo2.head(10)['model'].values
ConditionsAndConversionsModels = ConditionsAndConversionsModels[(ConditionsAndConversionsModels['model'].isin(arrayLabels))]
ConditionsAndConversionsModels = pd.crosstab(ConditionsAndConversionsModels.condition, \
                                   ConditionsAndConversionsModels.model)
#plot
fig,ax = plt.subplots(figsize=(10,7))
g = sns.heatmap(ConditionsAndConversionsModels,  cmap="Oranges", ax=ax,cbar_kws={'label': 'Cantidades'})

plt.title('10  modelos con mas conversiones y mayor cantidad por condicion', fontsize = 18)
plt.ylabel('Conditions', fontsize = 18)
plt.xlabel('Models', fontsize = 18)
#Pasamos a datetime la columna timestamp
models['timestamp'] = pd.to_datetime(models['timestamp'], errors='raise')

#Creamos una columna de meses
models['months'] = models['timestamp'].dt.month

#Filtramos por conversiones
mesesVenta = models[(models['event'] == 'conversion')] 

#Miramos los meses de mayor venta y como se puede ver lidera el mes 5.
mesesVenta['months'].value_counts()
#Obtengo los mas vendidos en Mayo
masVendidosMayo = mesesVenta[(mesesVenta['months'] == 5)]
masVendidosMayo = masVendidosMayo.groupby('model')['event'].count().sort_values(ascending = False)

#Me quedo con los 6 primeros
masVendidosMayo = np.array(masVendidosMayo.head(6).index)
#Como se aprecia el dorado es el mas comprado
MVC = mesesVenta[mesesVenta['model'].isin(masVendidosMayo)].groupby('color')['model'].count().sort_values(ascending = False)
MVC
#Pequenio grafico
colores =MVC.index
cantidades = MVC.values
explode = (0.1, 0, 0, 0, 0, 0) 

fig1, ax1 = plt.subplots()

ax1.pie(cantidades, explode=explode, labels=colores,
        shadow=True, startangle=90, colors=['gold','gray','silver','k','w','pink'], pctdistance= 0.7)

ax1.set(aspect="equal", title='Colores mas relevantes (10 modelos con mas ventas)')

plt.show()
