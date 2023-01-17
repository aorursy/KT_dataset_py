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
import pandas as pd
import matplotlib.pyplot as plt
private_df=pd.read_csv('../input/us-schools-dataset/Private_Schools.csv')
public_df = pd.read_csv('../input/us-schools-dataset/Public_Schools.csv')
private_df.head()
# Lets plot the number of cities in each state 
private_cities_state= private_df[['STATE', 'CITY']].groupby('STATE').count().sort_values('CITY', ascending=False)
public_cities_state= public_df[['STATE', 'CITY']].groupby('STATE').count().sort_values('CITY', ascending=False)


col = ['yellow', 'green', 'red', 'blue']
public_cities_state.plot(kind = 'bar', legend=False, color=col, figsize=(10,5))
plt.show()
col = ['yellow', 'green', 'red', 'blue']
private_cities_state.plot(kind = 'bar', legend=False, color=col, figsize=(10,5))
plt.show()
private_df['VAL_METHOD'].value_counts()
# Lets plot the number of cities in each state 
Private_State_method= private_df[['STATE', 'VAL_METHOD']].groupby('STATE').count().sort_values('VAL_METHOD', ascending=False)
Public_State_method= public_df[['STATE', 'VAL_METHOD']].groupby('STATE').count().sort_values('VAL_METHOD', ascending=False)

print (State_method)

Private_State_method.plot(kind = 'bar', legend=False, color='blue', figsize=(10,5))
plt.show()
Public_State_method.plot(kind = 'bar', legend=False, color='blue', figsize=(10,5))
plt.show()
