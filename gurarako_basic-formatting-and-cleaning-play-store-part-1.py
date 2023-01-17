%matplotlib inline

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

plt.style.use('ggplot')



import seaborn as sns # for making plots with seaborn

color = sns.color_palette()

sns.set(rc={'figure.figsize':(25,15)})



import warnings

warnings.filterwarnings('ignore')
df1 = pd.read_csv('../input/googleplaystore.csv')

df1.head()
len(df1)
df1[df1.duplicated(subset='App', keep='first')].head()
# Example of duplicated apps

df1[df1['App']=='Google Ads']
# Remove any duplicates app and only keep the first one

df1.drop_duplicates(keep='first', subset='App', inplace=True)
len(df1)
df1[df1['Android Ver'].isnull()]
# delete row with index number 10472

df1.drop([10472], inplace=True)
len(df1)
df1.dtypes
# delete  '+' and ',' and convert into integer

df1['Installs'] = df1['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else x)

df1['Installs'] = df1['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else x)

df1['Installs'] = df1['Installs'].apply(lambda x: int(x))
# convert into integer

df1['Reviews'] = df1['Reviews'].apply(lambda x: int(x))
# delete '$' and convert into float 

df1['Price'] = df1['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))

df1['Price'] = df1['Price'].apply(lambda x: float(x))
df1['Size'] = df1['Size'].apply(lambda x: str(x).replace('Varies with device', 'NaN') if 'Varies with device' in str(x) else x)

df1['Size'] = df1['Size'].apply(lambda x: str(x).replace('M', '') if 'M' in str(x) else x)

df1['Size'] = df1['Size'].apply(lambda x: str(x).replace(',', '') if 'M' in str(x) else x)

df1['Size'] = df1['Size'].apply(lambda x: float(str(x).replace('k', '')) / 1000 if 'k' in str(x) else x)

df1['Size'] = df1['Size'].apply(lambda x: float(x))
df1.dtypes
df1.head()
# save the cleaned dataset 

df1.to_csv('playstoreapps_cleaned.csv',index=False)