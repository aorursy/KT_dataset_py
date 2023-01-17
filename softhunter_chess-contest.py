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
data=pd.read_csv('../input/chess/games.csv')
threshold = sum(data.turns) / len(data.turns)

print(threshold)

data["turns_level"]= [ "high" if i > threshold else "low" for i in data.turns]

print(data["turns_level"])

data.loc[:20,["turns_level" , "turns"]]
data_new=data.head()

data.tail()

data.shape

data.info()

print(data.columns)

print(data_new)
melted=pd.melt(frame=data_new , id_vars='id' , value_vars=['rated','victory_status' ])

print(melted)

melted.pivot(index='id' , columns='variable' , values='value')
data_1=data["victory_status"].head()

data_2=data["winner"].head()



tot_data=pd.concat([data_1 , data_2] , axis=1)

print(tot_data)
data["turns"]=data["turns"].astype("float")

data.dtypes
data["turns"].value_counts(dropna=False)

data1=data



data1["turns"].dropna(inplace=True)

print(data1)
assert data1["turns"].notnull().all() # if the code has not a null character , it don't return anything.