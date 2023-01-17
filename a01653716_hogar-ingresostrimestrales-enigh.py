# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Ingresos = pd.read_csv('/kaggle/input/conjunto_de_datos_ingresos_enigh_2018_ns.csv')
Ingresos
Total_de_ingresos = Ingresos[Ingresos['clave'] == 'P021']
Total_de_ingresos

import numpy as np
from scipy import stats
Ingreso_tri = Total_de_ingresos['ing_tri']
Varianza = np.var(Ingreso_tri)
Varianza
media =np.mean(Ingreso_tri)
media
S = np.std(Ingreso_tri)
S
moda =stats.mode(Ingreso_tri)
moda
import matplotlib.pyplot as plt