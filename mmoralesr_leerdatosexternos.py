# cambiar a la carpeta donde se tienen los datos

#os.chdir('/home/mario/Dropbox/cursos/python/haslwanter')

#print( os.path.abspath(os.curdir) )

# Listar los archivos en el directorio entre comillas 

#print(os.listdir('/home/mario/Dropbox/cursos/python/haslwanter') )

# cargar numpy 

import numpy as np

datos=np.loadtxt('datap44hlt.txt',delimiter=',')

print(datos,type(datos))

datos.shape
import pandas as pd

# cambiar a la carpeta donde se tienen los datos

#os.chdir('/home/mario/Dropbox/cursos/python/haslwanter')

# verificar cual es el directorio actual 

#print( os.path.abspath(os.curdir) )

# Listar los archivos en el directorio entre comillas 

#os.listdir('/home/mario/Dropbox/cursos/python/haslwanter') 
df=pd.read_csv('data.txt')

#print( df.head() ) 

#print( df.tail() )

#df.info()

#df.shape
df2=pd.read_csv('data.txt',header=None)

#df2.info()

#df2.head()

#df2.tail()

#df2.columns

df2.columns=['Col0','Col1','Col2','Col3','Col4']

df2.head()
# generar los nombres de columnas de forma automática 

import numpy as np

a=np.repeat("Col",5) 

#print(a)

b=np.arange(5) 

#print(b)

# Convierte b que es numérico a str  

b=np.char.array(b.astype(str))

#b=b.astype(str) 

#print(b)

# pega a con b 

nombres=a+b

#print(nombres)

df2.columns=nombres

df2.head() 
data=np.loadtxt('data.txt',delimiter=',')

#print(type(data))

data=pd.DataFrame(data)

#print(type(data))

data.head()
df3=pd.read_csv('data2.txt') 

df3.head()

#df3.tail()

df3.info()

#df.shape

#df3
df3['nivel']=df3['nivel'].astype('category')

#print(df3.info())

#print(df3.nivel.cat.categories)

df3.nivel=df3.nivel.cat.rename_categories({1.0: 'Nivel 1', 2.0: 'Nivel 2'})

#print(df3.nivel.cat.categories)

df3.nivel.value_counts()
#df3 = pd.read_csv('data3.txt',engine='python') # no trabaja 

df3 = pd.read_csv('data3.txt',skiprows=[0,1,2,3],skipfooter=3,engine='python')

df3.tail()

#?pd.read_csv
xls=pd.ExcelFile('datosExel.xls')

data=xls.parse('hoja1', index_col=None, na_values='NA')

#?pd.ExcelFile

data.head()

data.tail()

#data.info()
import matplotlib.pyplot as plt

import seaborn as sns

#sns.regplot('y','z',data=data)

#plt.show()

#data["x"].hist(bins=8)

plt.hist("x",bins=8,density=True, histtype='stepfilled',  facecolor='g', alpha=0.75,data=data)

plt.show()

data2 = pd.read_excel('datosExel.xls','hoja1',index_col=None, na_values=['NA'])

data2.head()
# http://mmorales.tk/python/datosexcel.zip

import io

import zipfile



import sys

if sys.version_info[0] == 3:

    from urllib.request import urlopen

else:

    from urllib import urlopen

    

def get_data(url,inFile,sheet):

    '''Extraer datos desde un archivo zip'''



    # get the zip-archive

    archive = urlopen(url).read()

    

    # make the archive available as a byte-stream

    # hace que el archivo esté disponible como una secuencia de bytes

    zipdata = io.BytesIO()

    zipdata.write(archive)



    # extract the requested file from the archive, as a pandas XLS-file

    myzipfile = zipfile.ZipFile(zipdata)

    xlsfile = myzipfile.open(inFile)



    # read the xls-file into Python, using Pandas



    # read the xls-file into Python, using Pandas, and return the extracted data

    xls = pd.ExcelFile(xlsfile)

    df  = xls.parse(sheet) 



    return df



url = 'http://mmorales.tk/python/datosexcel.zip'

sheet='Hoja1'

inFile = r'datos.xls'

df=get_data(url,inFile,sheet)

df.head()

#ex74=pd.read_clipboard()

#ex74.head()

#ex74.tail()
from scipy.io import loadmat 