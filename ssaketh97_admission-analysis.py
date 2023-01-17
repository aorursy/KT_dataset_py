#Add All the imports here

import pandas as pd



import os

print(os.listdir("../input"))
#load the data

df = pd.read_csv('../input/Admission_Predict.csv',index_col = 'Serial No.')

df.head()
df.info()
sum(df.duplicated())
df.describe()
df.hist(figsize =(10,10));
pd.plotting.scatter_matrix(df,figsize =(17,17));
df.plot(x = 'GRE Score', y = 'Chance of Admit ', kind='scatter');
df.plot(x = 'TOEFL Score', y = 'Chance of Admit ', kind='scatter');
df.plot(x = 'CGPA', y = 'Chance of Admit ', kind='scatter');
df['GRE Score'].plot(kind = 'box');
df['TOEFL Score'].plot(kind = 'box');
df['University Rating'].plot(kind = 'box');
df['SOP'].plot(kind = 'box');
df['LOR '].plot(kind = 'box');
df['CGPA'].plot(kind = 'box');
df['Research'].plot(kind = 'box');