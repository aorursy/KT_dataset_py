# http://mmorales.tk/python/DatosEjemplos.zip

import io

import zipfile

import pandas as pd

import sys

if sys.version_info[0] == 3:

    from urllib.request import urlopen

else:

    from urllib import urlopen

    

def get_datascv(url,inFile):

    '''Extraer datos desde un archivo zip'''



    # get the zip-archive

    archive = urlopen(url).read()

    

    # make the archive available as a byte-stream

    # hace que el archivo esté disponible como una secuencia de bytes

    zipdata = io.BytesIO()

    zipdata.write(archive)



    # extract the requested file from the archive, as a pandas XLS-file

    myzipfile = zipfile.ZipFile(zipdata)

    csvfile = myzipfile.open(inFile)



    # read the xls-file into Python, using Pandas



    # read the xls-file into Python, using Pandas, and return the extracted data

    df= pd.read_csv(csvfile)

    #df  = xls.parse(sheet) 



    return df



url = 'http://mmorales.tk/python/DatosEjemplos.zip'

#sheet='Hoja1'

inFile = r'Datos_2912.csv'

df=get_datascv(url,inFile)

df=df.drop(columns=['Unnamed: 0'])

#df.columns

df.head()
# para la tabla de frecuencias

import numpy as np

minimo=df.Estatura.min()

print("minimo", df.Estatura.min())

maximo=df.Estatura.max()

print("maximo", df.Estatura.max())

Ancho=(maximo-minimo)/8 # (hay 8 clases)

print(Ancho)

#si quieres darle los límites

#limites=np.arange(minimo,maximo+Ancho,Ancho)

#[154, 156.75 , 159.5 , 162.25, 165. ,  167.75, 170.5,176]

#print(limites)

# si no quieres darle los límites, en bins le das el numero de intervalos (8)

### ojo que las da ordenadas de la de mayor frecuencia a la de menor 

pd.cut(df['Estatura'],bins=8, include_lowest=True).value_counts()