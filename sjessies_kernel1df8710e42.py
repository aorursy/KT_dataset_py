# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import json 

from pandas.io.json import json_normalize

#Cargamos los datos del Siata apartir del archivo JSon (Load de Json File)

with open('/kaggle/input/datos-siata-calidad-del-aire/Datos_SIATA_Aire_pm25.json') as f:

    d = json.load(f)

    

#Normalizamos el archivo Json (Normalize de Json file)

siata = json_normalize(d)

siata.head(3)
# Desempaquetamos la Columna de daos  (unpack the Column datos)

raw_data = json_normalize(d,'datos',['codigoSerial','nombreCorto','nombre','latitud','longitud'])

raw_data.head(3)