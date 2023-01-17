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
import seaborn as sb
from matplotlib import pyplot as plt 
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go



data = pd.read_csv("/kaggle/input/indian-migration-history/IndianMigrationHistory1.3.csv")

data.info()
# Change the name of columns 
data = data.rename(columns ={'Migration by Gender Name' : 'Gender_Name', 'Country Dest Name' : 'Destination',
                             '1960 [1960]' : '1960', '1970 [1970]' : '1970', '1980 [1980]' : '1980',
                             '1990 [1990]' : '1990', '2000 [2000]': '2000' } )
# Delete unused columns
data.drop(['Country Origin Name', 'Country Origin Code', 'Migration by Gender Code', 'Country Dest Code'] ,
          inplace = True , axis = 1)
data.head(10)
data.isnull().sum()
data.duplicated().sum()
# some of field fill by .. then we change them to 0 
# also change data type from object to int 
columns_list = ['1960', '1970','1980','1990','2000']
data[columns_list] = np.where (data[columns_list] == '..' , 0 , data[columns_list])
data[columns_list] = data[columns_list].astype(int) 
data.info()
# calculate number of people who migrate to ohter country 
# also we have some row which all decade fields are 0 in them thus we delete all of them
data['sum'] = data[columns_list].sum(axis = 1)
data.drop(data[data['sum'] == 0].index, inplace = True)
data.shape
data.describe()
# data in dataset brings seperately by sex then as its shows down we have duplicated in 'Destination' columns
data['Destination'].duplicated().sum()
# you can find which countries are first destination for indain 
data_sum = data.groupby(data['Destination']).sum()
df_sum = pd.DataFrame(data_sum).reset_index()
df_sum.sort_values(by = 'sum' , ascending = False, inplace = True)
df_sum
# make a dataframe based on each decade, show each decadehow many people migrate totaly 
data.sort_values(by = 'sum' , ascending = False, inplace = True)
decad = {'1960' , '1970' , '1980', '1990', '2000'}
peryear = []
for i in decad:
    row = [i , data[i].sum()]
    peryear.append(row)
df_peryear = pd.DataFrame(peryear)
df_peryear.columns =['Decade' ,'Sum']
df_peryear.sort_values(by= 'Decade', inplace = True)
df_peryear
df_peryear.plot.bar(x = 'Decade' , y = 'Sum')
# you can find the first destination for migration based on sex
data_sex = data.groupby(['Gender_Name', 'Destination']).sum().sort_values(by = 'sum' , ascending = False)
data_sex.head(10)