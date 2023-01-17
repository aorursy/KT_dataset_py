# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
data = pd.read_csv("../input/train.csv")
data.info()
# Para reducir memoria de los datasets:
def reduce_memory(dataset):
    ds_tmp = dataset.copy()
    int_columns = ds_tmp.select_dtypes(include=[np.int8,np.int16,np.int32,np.int64]).columns.tolist()
    for col in int_columns:
        ds_tmp[col] = pd.to_numeric(arg=ds_tmp[col], downcast='integer')

    float_columns = ds_tmp.select_dtypes(include=[np.float32,np.float64]).columns.tolist()
    for col in float_columns:
        ds_tmp[col] = pd.to_numeric(arg=ds_tmp[col], downcast='float')
    return(ds_tmp)
data_new = reduce_memory(data)
data_new.info()
espacio_memoria_inicial = round(data.memory_usage().sum()/(1024*1024) ,1)
espacio_memoria_final = round(data_new.memory_usage().sum()/(1024*1024),1)
print("Podemos observar en este caso(competencia) real se ha reducido el tamaño del dataset de",espacio_memoria_inicial,"a", espacio_memoria_final,"MB")
print("El cual representa una reducción del {:0.2f} % del dataset original.".format(100*(espacio_memoria_inicial-espacio_memoria_final)/espacio_memoria_inicial))
