import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns                         

sns.set(color_codes = True)                   

%matplotlib inline  
uber_drives = pd.read_csv("../input/uber-drive-dataset-analysis/Uber Drive.csv")

uber_drives.head()
uber_drives.tail(10)
uber_drives.head(10)
uber_drives.shape
#Total number of elements = Rows * Columns  





1155*5
uber_drives.size
uber_drives.info()
uber_drives.isnull().head()
uber_drives.isnull().sum()
uber_drives.describe()
df = uber_drives.dropna()
df.head() 
df.info()
df.columns
df['START*'].unique()
uber_drives['START*'].nunique()
print("the Total number of Unique start destinations : ",uber_drives['START*'].nunique() )
uber_drives.columns
uber_drives['STOP*'].nunique()
print("the Total number of Unique  stop  destinations : ",uber_drives['STOP*'].nunique() )
uber_drives.loc[uber_drives["START*"] == "San Francisco", : ]
uber_drives.head(10)
uber_drives["START*"].value_counts().head()





# This is in descending order of occurences 
uber_drives["START*"].value_counts().max
uber_drives["STOP*"].value_counts().head()

df.head()
df['START*'].value_counts().head()
df['STOP*'].value_counts().head()
freq = df.groupby(["START*","STOP*"])

df["PURPOSE*"].unique()
uber_drives.head()
uber_drives_new = uber_drives.fillna('Not Mentioned ')



uber_drives_new.head()











ab = uber_drives_new.groupby(['CATEGORY*','PURPOSE*']).sum()



ab
type(ab)
ab.columns
ab['MILES*']
type(ab.columns)


plt.figure(figsize = (18,10))



ab['MILES*'].plot( kind= 'bar' )
uber_drives.head()
uber_drives_new = uber_drives.fillna('Not Mentioned ')



uber_drives_new.head()
uber_drives_new.groupby(['PURPOSE*']).sum()
uber_drives_new['MILES*'].sum()
uber_drives['MILES*'].sum()

uber_drives.head()
uber_drives['CATEGORY*'].value_counts()
plt.figure(figsize = (8,8))



sns.countplot(x = "CATEGORY*", data = uber_drives,  )
plt.figure(figsize = (17,8))



sns.countplot(x = "PURPOSE*", data = uber_drives,  )
uber_drives.head()
uber_drives["CATEGORY*"].unique()
uber_drives["CATEGORY*"].value_counts()
uber_drives.head(3)
uber_drives['MILES*'].sum()


ab = uber_drives_new.groupby(['CATEGORY*','PURPOSE*']).sum()
ab
ab['MILES*'].sum()
uber_drives_new.head()
uber_drives['CATEGORY*'].value_counts()
T = uber_drives['MILES*'].sum()



T
ab
B = 16.5+197+2089.5+508+911.7+2851.3+4389.3+523.7 



B
P = 15.1+180.2+18.2+504.2



P
T



#Total
print("Proportion of Business : ",(B/T)*100 )
print("Proportion of Personal : ",(P/T)*100 )