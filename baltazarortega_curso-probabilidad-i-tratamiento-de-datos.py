import pandas as pd
import matplotlib.pyplot as plt
def mediaAgrupada(numDatos, listaFrecuencias, listaMarca):
    sumatoria = 0
    rango = len(listaFrecuencias)
    for i in range(rango):
        sumatoria += listaFrecuencias[i] * listaMarca[i]
    return sumatoria / numDatos
import math
def desviacionEstandarAgrupado(numDatos, listaFrec, listaMarc, mediaAgrupada):
    sumatoria = 0
    rango = len(listaFrec)
    for i in range(rango):
        sumatoria += listaFrec[i] * (math.pow(listaMarc[i], 2))
    k = numDatos * (math.pow(mediaAgrupada, 2))
    sAlCuadrado = (sumatoria - k) / (numDatos - 1)
    raizS = math.sqrt(sAlCuadrado)
    return raizS
def varianzaAgrupado(numDatos, listaFrec, listaMarc, mediaAgrupada):
    sumatoria = 0
    rango = len(listaFrec)
    for i in range(rango):
        sumatoria += listaFrec[i] * (math.pow(listaMarc[i], 2))
    k = numDatos * (math.pow(mediaAgrupada, 2))
    varianza = (sumatoria - k) / (numDatos - 1)
    return varianza
ciudadesEU = ['AJO','AK-CHIN VILLAGE','AMADO','APACHE JUNCTION','ARI','ARIMATEA','ASH FORK','AVONDALE','AVRA VALLEY','BAGDAD','BENSON','BIG PARK','BISBEE','BITTER SPRINGS','BLACK CANYON CITY','BLACKWATER','BLUEWATER','BOUSE','BUCKEYE','BULLHEAD CITY','BURNSIDE','CAMERON','CAMP VERDE','CANYON DAY','CAREFREE','CASA GRANDE','CASAS ADOBES','CATALINA','CATALINA FOOTHILLS','CAVE CREEK','CENTRAL HEIGHTS-MIDLAND CITY','CHANDLER','CHILCHINBITO','CHINLE','CHINO VALLEY','CHUICHU','CIBECUE','CIBOLA','CLARKDALE','CLAYPOOL','CLIFTON','COLORADO CITY','CONGRESS','COOLIDGE','CORDES LAKES','CORNVILLE','CORONA DE TUCSON','COTTONWOOD','COTTONWOOD-VERDE VILLAGE','DENNEHOTSO','DESERT HILLS','DEWEY-HUMBOLDT','DILKON','DOLAN SPRINGS','DOUGLAS','DREXEL-ALVERNON','DREXEL HEIGHTS','DUDLEYVILLE','DUNCAN','EAGAR','EAST FORK','EAST SAHUARITA','EHRENBERG','ELGIN','EL MIRAGE','ELOY','FIRST MESA','FLAGSTAFF','FLORENCE','FLOWING WELLS','FORT DEFIANCE','FORTUNA FOOTHILLS','FOUNTAIN HILLS','FREDONIA','GADSDEN','GANADO','GILA BEND','GILBERT','GISELA','GLENDALE','GLOBE','GOLD CAMP','GOLDEN VALLEY','GOODYEAR','GRAND CANYON VILLAGE','GREASEWOOD','GREEN VALLEY','GUADALUPE','HAYDEN','HEBER-OVERGAARD','HOLBROOK','HOTEVILLA-BACAVI','HOUCK','HUACHUCA CITY','JEDDITO','JEROME','KACHINA VILLAGE','KAIBAB','KAIBITO','KAYENTA','KEAMS CANYON','KEARNY','KINGMAN','KYKOTSMOVI VILLAGE','LAKE HAVASU CITY']
ciudades = ciudadesEU[:45]
vientos = [8.9, 7.1, 9.1, 8.8, 10.2, 12.4, 11.8, 10.9, 12.7, 10.3, 
          8.6, 10.7, 10.3, 8.4, 7.7, 
          11.3, 7.6, 9.6, 7.8, 10.6, 
          9.2, 9.1, 7.8, 5.7, 8.3, 
          8.8, 9.2, 11.5, 10.5, 8.8,
          35.1, 8.2, 9.3, 10.5, 9.5, 
          6.2, 9.0, 7.9, 9.6, 8.8, 
          7.0, 8.7, 8.8, 8.9, 9.4]
datos = dict(zip(ciudades, vientos))
serie = pd.Series(datos)
serie
serie.mean()
serie.median()
moda = serie.mode()
serie.value_counts().head()
# Varianza
serie.var()
# Desviacion estadar: 4.13
serie.describe()
# Frecuencia absoluta
serie.value_counts(bins=6, ascending=True)

# Media agrupada
marca = [8.15, 13.15, 18.15, 23.15, 28.15, 33.15]
frec = [37, 7, 0, 0, 0, 1]
resMediaAgrupada = mediaAgrupada(45, frec, marca)
print(resMediaAgrupada)
varianzaAgrupada = varianzaAgrupado(45, frec, marca, resMediaAgrupada)
varianzaAgrupada
# Desviacion estandar agrupada
stdAgrupada = desviacionEstandarAgrupado(45, frec, marca, resMediaAgrupada)
print(stdAgrupada)


plt.xlabel('Velocidad del viento mi/h')
plt.ylabel('Cantidad de ciudades')
plt.hist(vientos, bins=[5.7, 10.6, 15.6, 20.6, 25.6, 30.6, 35.6], rwidth=0.95)
ciudades_v2 = ciudadesEU[:44]
vientos_v2 = [8.9, 7.1, 9.1, 8.8, 10.2, 12.4, 11.8, 10.9, 12.7, 10.3, 
          8.6, 10.7, 10.3, 8.4, 7.7, 
          11.3, 7.6, 9.6, 7.8, 10.6, 
          9.2, 9.1, 7.8, 5.7, 8.3, 
          8.8, 9.2, 11.5, 10.5, 8.8, 8.2, 9.3, 10.5, 9.5, 
          6.2, 9.0, 7.9, 9.6, 8.8, 
          7.0, 8.7, 8.8, 8.9, 9.4]
datos_v2 = dict(zip(ciudades_v2, vientos_v2))
serie_v2 = pd.Series(datos_v2)
len(serie_v2)
serie_v2.mean()
serie_v2.median()
moda = serie_v2.mode()
serie_v2.value_counts().head()
# Varianza
serie_v2.var()
# Desviacion estadar: 1.51
serie_v2.describe()
# Frecuencia absoluta
serie_v2.value_counts(bins=[5.7, 7.1, 8.5, 10, 11.5, 13], ascending=True)
# Media agrupada
marca = [6.4, 7.85, 9.3, 10.8, 12.3]
frec = [4, 9, 18, 10, 3]
resMediaAgrupada = mediaAgrupada(44, frec, marca)
print(resMediaAgrupada)
varianzaAgrupada = varianzaAgrupado(44, frec, marca, resMediaAgrupada)
varianzaAgrupada
# desviacion estandar agrupada

algo = desviacionEstandarAgrupado(44, frec, marca, resMediaAgrupada)
print(algo)

plt.xlabel('Velocidad del viento mi/h') 
plt.ylabel('Cantidad de ciudades') 
plt.hist(vientos_v2, bins=[5.7, 7.1, 8.5, 10, 11.5, 13], rwidth=0.95, color='green')

muestras = [.74, .32, 1.66, 3.59, 4.55,
           6.47, 9.99, .70, .37, .76, 
           1.9, 1.77, 2.42, 1.09, 2.03, 
           2.69, 2.41, .54, 8.52, 5.7, 
           .75, 1.96, 3.36, 4.06, 12.48]
serie_3 = pd.Series(muestras)
len(serie_3)
media = serie_3.mean()
media
serie_3.median()
moda = serie_3.mode()
serie_3.value_counts().head()
# Varianza
serie_3.var()
# Desviacion estadar: 3.18
serie_3.describe()
# Frecuencia absoluta
serie_3.value_counts(bins=[.32, 2.75, 5.19, 7.63, 10.07, 12.51], ascending=True)
# Media agrupada
marca = [1.53, 3.97, 6.41, 8.85, 11.29]
frecuencia = [16, 4, 2, 2, 1]
resMediaAgrupada = mediaAgrupada(25, frecuencia, marca)
print(resMediaAgrupada)
varianzaAgrupada = varianzaAgrupado(25, frecuencia, marca, resMediaAgrupada)
varianzaAgrupada
# desviacion estandar agrupada

resStdAgrupada = desviacionEstandarAgrupado(25, frecuencia, marca, resMediaAgrupada)
print(resStdAgrupada)

plt.xlabel('Cantidad de material radioactivo') 
plt.ylabel('Frecuencia') 
plt.hist(muestras, bins=[.32, 2.75, 5.19, 7.63, 10.07, 12.51], rwidth=0.95, color='yellow')
acciones = [11.88, 7.99, 7.15, 7.13, 6.27,
           6.07, 5.98, 5.91, 5.49, 5.26, 
           5.07, 4.94, 4.81, 4.79, 4.55, 
           4.43, 4.40, 4.05, 3.94, 3.93, 
           3.78, 3.69, 3.62, 3.48, 3.44, 
           3.36, 3.26, 3.20, 3.11, 3.03, 
           2.99, 2.89, 2.88, 2.74, 2.74, 
           2.69, 2.68, 2.63, 2.62, 2.61]
serie_4 = pd.Series(acciones)

serie_4.mean()
serie_4.median()
serie_4.value_counts().head()
serie_4.var()
# Desviacion estandar: 1.87
serie_4.describe()

# Frecuencia absoluta
serie_4.value_counts(bins=[2.61, 4.46, 6.32, 8.18, 10.04, 11.9], ascending=True)
# Media agrupada
marca = [3.53, 5.39, 7.25, 9.11, 10.97]
frecuencia = [25, 11, 3, 0, 1]
resMediaAgrupada = mediaAgrupada(40, frecuencia, marca)
print(resMediaAgrupada)
varianzaAgrupada = varianzaAgrupado(40, frecuencia, marca, resMediaAgrupada)
varianzaAgrupada
# desviacion estandar agrupada

resStdAgrupada = desviacionEstandarAgrupado(40, frecuencia, marca, resMediaAgrupada)
print(resStdAgrupada)
plt.xlabel('porcentaje de acciones') 
plt.ylabel('Frecuencia') 
plt.hist(acciones, bins=[2.61, 4.46, 6.32, 8.18, 10.04, 11.9], rwidth=0.95, color='pink')
