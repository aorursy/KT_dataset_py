

import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier 
import matplotlib.pyplot as plt  
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns  #importing packages for visualization

df = pd.read_csv('../input/us-counties-covid-19-dataset/us-counties.csv') 
df.info()

df.head()

df['date'] =pd.to_datetime(df['date'])
df.info()

df['county'].value_counts().head(10)

df['deaths'].value_counts().head(5)
print('Max deaths',df['deaths'].max())
print('Min deaths',df['deaths'].min())
print('Avg deaths',df['deaths'].mean())

df['cases'].value_counts().head(5)
print('Max Cases',df['cases'].max())
print('Min Cases',df['cases'].min())
print('Avg Cases',df['cases'].mean())

df[df['deaths']== df['deaths'].max()] 

df[df['cases']== df['cases'].max()]
df.hist()