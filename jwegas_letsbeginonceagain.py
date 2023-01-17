# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv', chunksize=1000000)
Semana_max = 0
Agencia_ID_max = 0
Canal_ID_max = 0
Ruta_SAK_max = 0
Cliente_ID_max = 0
Producto_ID_max = 0
Venta_uni_hoy_max = 0
Venta_hoy_max = 0
Dev_uni_proxima_max = 0
Dev_proxima_max = 0
Demanda_uni_equil_max = 0

for chunk in train:
    Semana_max = max([Semana_max, chunk['Semana'].max()])
    Agencia_ID_max = max([Agencia_ID_max, chunk['Agencia_ID'].max()])
    Canal_ID_max = max([Canal_ID_max, chunk['Canal_ID'].max()])
    Ruta_SAK_max = max([Ruta_SAK_max, chunk['Ruta_SAK'].max()])
    Cliente_ID_max = max([Cliente_ID_max, chunk['Cliente_ID'].max()])
    Producto_ID_max = max([Producto_ID_max, chunk['Producto_ID'].max()])
    Venta_uni_hoy_max = max([Venta_uni_hoy_max, chunk['Venta_uni_hoy'].max()])
    Venta_hoy_max = max([Venta_hoy_max, chunk['Venta_hoy'].max()])
    Dev_uni_proxima_max = max([Dev_uni_proxima_max, chunk['Dev_uni_proxima'].max()])
    Dev_proxima_max = max([Dev_proxima_max, chunk['Dev_proxima'].max()])
    Demanda_uni_equil_max = max([Demanda_uni_equil_max, chunk['Demanda_uni_equil'].max()])
    
Cliente_ID_max
train_100 = pd.read_csv('../input/train.csv', nrows=100)
train_100.head()