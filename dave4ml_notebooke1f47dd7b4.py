# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

df = pd.read_csv("../input/train_ver2.csv", parse_dates=['fecha_dato', 'fecha_alta'], low_memory=False, nrows =500000)

df.head()

#df.info()



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
cust = df['ncodpers']

month= df['fecha_dato'].apply(lambda x: x.month)

cols = ["fecha_dato", "ind_empleado","pais_residencia","sexo","age", "fecha_alta", "ind_nuevo","antiguedad","indrel","ult_fec_cli_1t","indrel_1mes","tiprel_1mes","indresi","indext","conyuemp","canal_entrada","indfall","tipodom","cod_prov","nomprov","ind_actividad_cliente","renta","segmento"]

char = df[cols]

frames = [cust, month, char]

data = pd.concat(frames, axis = 1)

data.fillna('NA')

data.head()