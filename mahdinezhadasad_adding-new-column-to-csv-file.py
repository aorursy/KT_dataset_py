# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/pokemon-challenge/pokemon.csv')
data.info()
data.describe()
data.head()
Type_new = pd.Series([]) 

  

for i in range(len(data)): 

    if data["Type 1"][i] == "Grass": 

        Type_new[i]="Green"

  

    elif data["Type 1"][i] == "Fire": 

        Type_new[i]="Orange"

  

    elif data["Type 1"][i] == "Water": 

        Type_new[i]="Blue"

  

    else: 

        Type_new[i]= data["Type 1"][i] 

  

          

# inserting new column with values of list made above         

data.insert(2, "Type New", Type_new) 

  

# list output 

data.head() 
data.drop('Type 1',axis=1, inplace=True)
data.head(30)
data.rename(columns={'Date': 'date', 

                     'Id': 'id',

                     'Province/State':'state',

                     'Country/Region':'country',

                     'Lat':'lat',

                     'Long': 'long',

                     'ConfirmedCases': 'confirmed',

                     'Fatalities':'deaths',

                    }, inplace=True)

data.head()
data=data.replace('2/14/2020',0 )

data=data.replace('2/26/2020',0 )

data=data.replace('2/13/2020',0 )

data=data.replace('2/28/2020',0 )

data=data.replace('2/26/2020',0 )

data=data.replace('2/27/2020',0 )

data=data.replace('2/25/2020',0 )

data=data.replace('2/23/2020',0 )

data=data.replace('2/22/2020',0 )

data=data.replace('2/24/2020',0 )

data=data.replace('2/1/2020',0 )

data=data.replace('2/19/2020',0 )

data=data.replace('2/21/2020',0 )

data=data.replace('02/01/20',0 )

data["death"] =data["death"].astype(int) 

Data_per_country = data.groupby(["country"])["death"].sum().reset_index().sort_values("death",ascending=False).reset_index(drop=True)
Data_per_country