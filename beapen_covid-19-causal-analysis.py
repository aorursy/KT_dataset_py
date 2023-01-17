# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install causalnex > null
!pip install pandas==0.25.0 > null

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_excel('/kaggle/input/covid19-open-access-data/COVID19_2020_open_line_list.xlsx', sheet_name='Hubei')
data = data[['age', 'sex', 'city', 'latitude', 'longitude', 'symptoms', 'reported_market_exposure', 'chronic_disease_binary', 'outcome' ]]

data.describe()
data["outcome"].fillna("alive", inplace = True) 

data["reported_market_exposure"].fillna("no", inplace = True)

data["chronic_disease_binary"].fillna("no", inplace = True)

data["symptoms"].fillna("none", inplace = True)

data = data.dropna(axis = 0, how ='any') 

data.head(10)
struct_data = data.copy()



non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)

print(non_numeric_columns)
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

for col in non_numeric_columns:

    struct_data[col] = le.fit_transform(struct_data[col].astype(str))



struct_data.describe()
from causalnex.structure.notears import from_pandas

sm = from_pandas(struct_data)
from causalnex.plots import plot_structure

sm.remove_edges_below_threshold(0.8)

sm = sm.get_largest_subgraph()

_, _, _ = plot_structure(sm)
data = pd.read_excel('/kaggle/input/covid19-open-access-data/COVID19_2020_open_line_list.xlsx', sheet_name='outside_Hubei')



data = data[['age', 'sex', 'city', 'latitude', 'longitude', 'symptoms', 'reported_market_exposure', 'chronic_disease_binary', 'outcome' ]]

data.describe()

data["outcome"].fillna("alive", inplace = True) 

data["reported_market_exposure"].fillna("no", inplace = True)

data["chronic_disease_binary"].fillna("no", inplace = True)

data["symptoms"].fillna("none", inplace = True)

data = data.dropna(axis = 0, how ='any') 





struct_data = data.copy()



non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)



from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

for col in non_numeric_columns:

    struct_data[col] = le.fit_transform(struct_data[col].astype(str))



struct_data.head(10)
struct_data.describe()

struct_data['outcome'].value_counts()
from causalnex.structure.notears import from_pandas

sm = from_pandas(struct_data)



from causalnex.plots import plot_structure

sm.remove_edges_below_threshold(0.8)

sm = sm.get_largest_subgraph()

_, _, _ = plot_structure(sm)