# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

!ls ../input/



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from IPython.display import Image

Image("../input/img-nan/nan.jpg")
# Leyendo con delimitador por defecto ","

pd.read_csv('/kaggle/input/pruebanan/pruebaNaN.csv')
# Leyendo con delimitador ";"

# Por defecto los valores keep_default_na=True 

pd.read_csv('/kaggle/input/pruebanan/pruebaNaN.csv' ,delimiter=";")
# Comprobando que valores son nulos (nan), todos menos el de col4

pd.read_csv('/kaggle/input/pruebanan/pruebaNaN.csv' ,delimiter=";").isnull()
# Leyendo con delimitador ";"

# Por defecto los valores keep_default_na=False 

pd.read_csv('/kaggle/input/pruebanan/pruebaNaN.csv', keep_default_na=False, delimiter=";")
# Comprobando que valores son nulos (nan), ninguno...

pd.read_csv('/kaggle/input/pruebanan/pruebaNaN.csv', keep_default_na=False, delimiter=";").isnull()
# Por defecto los valores keep_default_na=False y 

# además le indico que solo van a ser valores nulos, los vacios (espacio en blanco)!

pd.read_csv('/kaggle/input/pruebanan/pruebaNaN.csv', keep_default_na=False, delimiter=";", na_values=[""])
Image("../input/img-nan/nan.jpg")
# Efectivamente el unico valor nulo es el primer elemento de la col6 que era vacio inicialmente.

pd.read_csv('/kaggle/input/pruebanan/pruebaNaN.csv', keep_default_na=False, delimiter=";", na_values=[""]).isnull()