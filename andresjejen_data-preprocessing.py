# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Secop = pd.read_csv("/kaggle/input/datatonanticorrupcionbeitlab/SECOP_I_FilterAdiciones5M.csv") 

# Preview the first 5 lines of the loaded data 

Secop.head()
Secop.info()
Municipios = pd.read_csv("/kaggle/input/datatonanticorrupcionbeitlab/municipiosFree5.csv") 

# Preview the first 5 lines of the loaded data 

Municipios.head()
Multas = pd.read_csv("/kaggle/input/datatonanticorrupcionbeitlab/Multas_y_Sanciones_SECOP_I.csv") 

# Preview the first 5 lines of the loaded data 

Multas.head()
Result1 = pd.merge(Multas,Municipios,left_on="Municipio", right_on='Code',how='left') 
Result1.head()
Result1.info()
Result = pd.merge(Secop,Result1,left_on="Ruta Proceso en SECOP I", right_on="Ruta de Proceso",how='outer') 
Result.head()
Result.info()