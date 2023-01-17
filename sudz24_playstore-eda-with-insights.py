import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#To hide Warning messages.
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("../input/googleplaystore.csv")

df.head()
print(df.isnull().sum())

df.dropna(inplace=True) #Dropping Rows with Null values

df.drop_duplicates(inplace=True)
df.shape
df.dtypes  # Displaying Data types of each feature.
df.Reviews = df.Reviews.astype('int64') #Changing to int type.
newInstalls = []

for row in df.Installs:
    
    row = row[:-1]
    newRow = row.replace(",", "")
    newInstalls.append(float(newRow))
    

df.Installs = newInstalls

df.Installs.head()
newSize = []

for row in df.Size:
    newrow = row[:-1]
    try:
        newSize.append(float(newrow))
    except:
        newSize.append(0) #When it says - Size Varies.
    
df.Size = newSize

df.Size.head()
newPrice = []

for row in df.Price:
    if row!= "0":
        newrow = float(row[1:])
    else:
        newrow = 0 
        
    newPrice.append(newrow)
        
df.Price = newPrice

df.Price.head()
    
newVer = []

for row in df['Android Ver']:
    try:
        newrow = float(row[:2])
    except:
        newrow = 0  # When the value is - Varies with device
    
    newVer.append(newrow)
    
df['Android Ver'] =  newVer

df['Android Ver'].value_counts()
df.Category.value_counts() 
df.Category.value_counts().plot(kind='barh',figsize= (12,8))
df.Rating.describe()
sns.distplot(df.Rating)
print("No. of Apps with full ratings: ",df.Rating[df['Rating'] == 5 ].count())
plt.figure(figsize=(10,5))
sns.distplot(df.Reviews)
df[df.Reviews>40000000]
plt.pie(df.Type.value_counts(), labels=['Free', 'Paid'], autopct='%1.1f%%')
df[df.Price == df.Price.max()]
df['Android Ver'].value_counts()
sns.countplot(df['Android Ver'])
df_full = df[df.Rating == 5]

df_full.head()
sns.distplot(df_full.Installs)
df_full.Installs.value_counts().sort_index()
df_full_maxinstalls = df_full[df.Installs > 1000]

df_full_maxinstalls[['App', 'Category', 'Installs']]
sns.distplot(df_full.Reviews)
df_full = df_full[df.Reviews > 30]
print("No. of Apps having 5.0 Rating with sufficient Reviews: ",df_full.App.count())
plt.figure(figsize=(12,5))
sns.countplot(df_full.Genres)

sns.countplot(df_full.Price)