#Importación de las librerías de interés

import numpy as np                # linear algebra

import pandas as pd               # data frames

import seaborn as sns             # visualizations

import matplotlib.pyplot as plt   # visualizations

import scipy.stats                # statistics



from scipy.cluster.hierarchy import dendrogram, linkage

from scipy.cluster.hierarchy import complete, fcluster #Cluster Jerarquico Completo



from sklearn import preprocessing

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA



import os

print(os.listdir("../input"))
#Se renombra la base de datos a trabajar y se realiza la exploración inicial

df = pd.read_csv("../input/heart.csv")



# Print the head of df

print(df.head())

# Print the info of df

print(df.info())

# Print the shape of df

print(df.shape)
#Médidas básicas para todas las variables

df.describe()
#¿La base está balaceada?

df.isnull().sum()
#histograma para variables continuas

f, axes = plt.subplots(2,2, figsize=(20, 12))

f.suptitle("Distribución variables continuas", fontsize=20)

sns.distplot( df["age"], ax=axes[0,0])

sns.distplot( df["chol"], ax=axes[0,1])

sns.distplot( df["thalach"], ax=axes[1,0])

sns.distplot( df["oldpeak"], ax=axes[1,1])
#Histogramas para las variables categoricas 

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(18, 10))

fig.suptitle("Distribución de frecuencias para las variables categoricas", fontsize=20)



x = ['Mujer','Hombre'];y = df.sex.value_counts(sort = False).values

axes[0][0].bar(x,y);axes[0][0].set_title('Sexo')



x = ['Typical','Atypical','Non-Anginal','Aysyptomatic'];y = df.cp.value_counts(sort = False).values

axes[0][1].bar(x,y);axes[0][1].set_title('Dolor en el Pecho-cp')



x = ['<120 mg/dl','>120 mg/dl'];y = df.fbs.value_counts(sort = False).values

axes[0][2].bar(x,y);axes[0][2].set_title('Azúcar sangre ayunas-fbs')



x = ['Regular','Abnormality','Severe'];y = df.restecg.value_counts(sort = False).values

axes[0][3].bar(x,y);axes[0][3].set_title('Electrocardiograma-restecg')



x = ['No','Si'];y = df.exang.value_counts(sort = False).values

axes[1][0].bar(x,y);axes[1][0].set_title('Angina por ejercicio-exang')



x = ['Downward','Flat','Upward'];y = df.slope.value_counts(sort = False).values

axes[1][1].bar(x,y);axes[1][1].set_title('ST excercise peak-slope')



x = ['None','Normal','Fixed Defect','Reversable Defect'];y = df.thal.value_counts(sort = False).values

axes[1][2].bar(x,y);axes[1][2].set_title('Thalium Stress Test - Thal')



x = ['No','Si'];y = df.target.value_counts(sort = False).values

axes[1][3].bar(x,y);axes[1][3].set_title('Enfermedad Coronaria - Target')



plt.show()
#los nombres no dicen mucho, entonces se renombraran

df=df.rename(columns={'age':'Edad','sex':'Sexo','cp':'Dolor_pecho','trestbps':'Presión','chol':'Colesterol','fbs':'Azúcar >120','restecg':'Electro','thalach':'Ritmo_cardiaco','exang':'Angina_ejercicio','oldpeak':'Depresión_ST ','slope':'Pico_ejercicio','ca':'Vasos_Ppales','thal':'Defecto','target':'Diagnostico'})

df.columns
# Compute the correlation matrix

corr=df.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool);mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
#Grafico de violin

f, axes = plt.subplots(1,2, figsize=(20, 12))

f.suptitle("Ritmo cardiaco Vs. Prueba esfuerzo y tipo de dolor de pecho", fontsize=20)



sns.catplot(x="Pico_ejercicio", y="Ritmo_cardiaco", hue="Diagnostico",kind="violin", split=True, data=df,ax=axes[0] )

sns.catplot(x="Dolor_pecho", y="Ritmo_cardiaco", hue="Diagnostico",kind="violin", split=True, data=df, ax=axes[1] ) 
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(18, 10))

fig.suptitle("Ritmo cardiaco Vs. Prueba esfuerzo y tipo de dolor de pecho", fontsize=20)



sns.catplot(x="Dolor_pecho", y="Ritmo_cardiaco", hue="Diagnostico",kind="point", ax=axes[0],data=df )

sns.catplot(x="Pico_ejercicio", y="Ritmo_cardiaco", hue="Diagnostico",kind="point", data=df,ax=axes[1])
#Codificando las variables categoricas

df = pd.read_csv("../input/heart.csv")

#creamos variables categoricas separadas y quitamos una para no generar problemas de multicolinealidad

df_dum=pd.get_dummies(df,columns=["sex","cp","fbs","restecg","exang","slope","thal","target"],drop_first=True)

df_dum.describe()
heart_scale = df_dum

scaler = preprocessing.StandardScaler()

columns =df_dum.columns

heart_scale[columns] = scaler.fit_transform(heart_scale[columns])

heart_scale.head()
#Primero generamos un dendograma y generamos la matriz de distancias "Z" que se requiere para 

#el cluster jerarquico completo



Z = linkage(heart_scale.loc[:,["thalach","cp_1","cp_2","cp_3","target_1","slope_1","slope_2"]], 'complete', metric='euclidean')

fig = plt.figure(figsize=(25, 10))

dn = dendrogram(Z)

plt.show()

#Pasamos a generar los cluster, para lo cual definimos K=5 y K=3

#conforme a la información del dendograma



h_cluster=df.copy()

h_cluster['5_clust']=fcluster(Z,t=5, criterion='distance')

h_cluster['3_clust']=fcluster(Z,t=3, criterion='distance')

h_cluster.head()
# Compute the correlation matrix

corr_cluster=h_cluster.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr_cluster, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_cluster, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})



#los nombres no dicen mucho, entonces se renombraran

h_cluster=h_cluster.rename(columns={'age':'Edad','sex':'Sexo','cp':'Dolor_pecho','trestbps':'Presión','chol':'Colesterol','fbs':'Azúcar >120','restecg':'Electro','thalach':'Ritmo_cardiaco','exang':'Angina_ejercicio','oldpeak':'Depresión_ST ','slope':'Pico_ejercicio','ca':'Vasos_Ppales','thal':'Defecto','target':'Diagnostico'})

h_cluster.columns



fig, axes = plt.subplots(1,2, figsize=(18, 10))

fig.suptitle("grupos de pancientes", fontsize=20)



sns.catplot(x="Dolor_pecho", y="Ritmo_cardiaco", hue="Diagnostico",col="5_clust", aspect=.6,

            kind="swarm", data=h_cluster);



sns.catplot(x="Pico_ejercicio", y="Ritmo_cardiaco", hue="Diagnostico",col="5_clust", aspect=.6,

            kind="swarm", data=h_cluster);

#Primero generamos un dendograma y generamos la matriz de distancias "Z" que se requiere para 

#el cluster jerarquico sencillo



Z = linkage(heart_scale.loc[:,["thalach","cp_1","cp_2","cp_3","target_1","slope_1","slope_2"]],

            'single', metric='euclidean')

fig = plt.figure(figsize=(25, 10))

dn = dendrogram(Z)

plt.show()
#Pasamos a generar los cluster, para lo cual definimos K=3

#conforme a la información del dendograma



h_single=df.copy()

h_single['3_clust']=fcluster(Z,t=5, criterion='distance')



h_single.head()
# Compute the correlation matrix

corr_single=h_single.corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr_single, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr_single, mask=mask, cmap=cmap, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})