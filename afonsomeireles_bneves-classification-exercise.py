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
import numpy as np  #algebra
import pandas as pd  #processamento de dados
import matplotlib.pyplot as plt  #plotagem
import seaborn as sns   #visualizações
sns.set(style="darkgrid")
data=pd.read_excel('/kaggle/input/bneves/Bneves1.xlsx')
test_data = pd.read_excel('/kaggle/input/bneves/Bneves_novos1.xlsx')
data = data.head(1000)
data.head()
data.info()
# converter os dados numéricos para categóricos
col_names = list(data) #retirar os nomes dos atributos
for col in col_names:
    data[col] = data[col].astype('category',copy=False)
    
data.info()
# adicionar categoria para substituir nos NULLS
data['Service'] = data['Service'].cat.add_categories('NaN')
data['Source Port'] = data['Source Port'].cat.add_categories('NaN')
data['Service'].fillna('NaN', inplace =True)
data['Source Port'].fillna('NaN', inplace =True)
data.info()


