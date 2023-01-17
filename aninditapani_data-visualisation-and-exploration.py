import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/core_dataset.csv')
df.head(7) #first seven data points
df.shape #(rows,columns)
df['Sex'].unique()
df['Sex'].replace('male','Male',inplace=True)
df.dropna(subset=['Sex'],inplace=True) # remove rows with nan values for Sex
df.shape
df['Sex'].value_counts() #Lets plot this
import matplotlib.pyplot as plt

df['Sex'].value_counts().plot(kind='bar')
# More female employees!
# Gender diversity across departmets
import seaborn as sns
plt.figure(figsize=(16,9))
ax=sns.countplot(x=df['Department'],hue=df['Sex'])
df['MaritalDesc'].value_counts().plot(kind='pie')
df['CitizenDesc'].unique()
df['CitizenDesc'].value_counts().plot(kind='bar')
df['Position'].unique()
plt.figure(figsize=(16,9))
df['Position'].value_counts().plot(kind='bar')
df['Pay Rate'].describe()
df['Age'].describe()
df.plot(x='Age',y='Pay Rate',kind='scatter')
# Looks like thery are not related! 
df['Performance Score'].isna().any()
df_perf = pd.get_dummies(df,columns=['Performance Score'])
df_perf.head(7)
col_plot= [col for col in df_perf if col.startswith('Performance')]
col_plot
fig, axes = plt.subplots(3, 3, figsize=(16,9))
for i,j in enumerate(col_plot):
    df_perf.plot(x=j,y='Pay Rate',ax = axes.flat[i],kind='scatter')
    
#Doesn't look like 
df['Manager Name'].unique()
plt.figure(figsize=(20,20))
sns.countplot(y=df['Manager Name'], hue=df['Performance Score'])
df['Pay Rate'].describe()
df.groupby('Department')['Pay Rate'].sum().plot(kind='bar')
#Production department pays more!
plt.figure(figsize=(16,9))
df.groupby('Position')['Pay Rate'].sum().plot(kind='bar')
df.loc[df['Pay Rate'].idxmax()]
# The CEO :p 