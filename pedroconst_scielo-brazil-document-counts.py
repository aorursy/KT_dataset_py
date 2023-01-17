import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline
import os
print(os.listdir("../input/tabs_bra"))
df = pd.read_csv('../input/tabs_bra/documents_counts.csv')
df.info()
df.head()
df.describe()
df['title thematic areas'].nunique()
df['title at SciELO'].nunique()
df['title current status'].unique()
df['document type'].nunique()
df['document publishing year'].nunique()
df['extraction date'].nunique()
plt.figure(figsize=(6,10))
plt.rc('axes', labelsize=12) 
sns.countplot(y='title thematic areas',hue='title current status',data=df,palette='viridis', dodge=False)
plt.figure(figsize=(10,6))
plt.rc('axes', labelsize=16) 
sns.countplot(y='document type',hue='title current status',data=df,palette='viridis', dodge=False)
plt.figure(figsize=(18,8))
plt.xticks(rotation=45)
plt.rc('axes', labelsize=22) 
sns.countplot(x='document publishing year',data=df[df['document publishing year']>=1970],hue='title current status',palette='viridis', dodge=False)
sns.pairplot(df[(df['authors']<15) & (df['pages']<50)][['authors','pages','references']])
sns.distplot(df[df['document type']=='research-article']['pages'])
sns.distplot(df[df['authors']<15]['authors'])
