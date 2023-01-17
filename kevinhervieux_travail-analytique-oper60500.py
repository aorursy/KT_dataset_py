# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
print("Setup Complete")
#Télécharger les données
my_filepath='../input/sales-forecasting/train.csv'
my_data = pd.read_csv(my_filepath,index_col='Order ID')
my_data


#Mettre les date (string) en date
pd.to_datetime(my_data['Order Date']) #marche pas
my_data['Order Date']
#modification de my_data

my_data['Délai entre commande et envoie']= my_data['Ship Date']-my_data['Order Date']

my_data.head(10)
# tableau de .....
sns.barplot(data=my_data,x='Region',y='Sales')

# tableau de bord
#profile = ProfileReport(my_data,
#                        title='Pandas Profiling Report',
#                        html={'style':{'full_width':True}})
#profile.to_widgets()
#profile.to_notebook_itframe()
