import numpy as np 
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
for x in glob('../input/*.csv'):
    df=pd.read_csv(x)
df.head()
print(df['postdate'].min())
print(df['postdate'].max())
mypalette = sns.color_palette('GnBu_d', 40)
plt.figure(figsize=(20,10))
sns.countplot(y=df['company'], order=df['company'].value_counts().index, palette=mypalette)
plt.ylabel('Company Name', fontsize=14)
plt.xlabel('Number of Job postings', fontsize=14)
plt.title("Companies with most job postings", fontsize=18)
plt.ylim(20.5,-0.5)
plt.show()
plt.figure(figsize=(20,10))
sns.countplot(y=df['jobtitle'], order=df['jobtitle'].value_counts().index, palette=mypalette)
plt.ylabel('Job Title', fontsize=14)
plt.xlabel('Number of Job postings', fontsize=14)
plt.title("Most seeked Jobs", fontsize=18)
plt.ylim(20.5,-0.5)
plt.show()
java_count = df['jobtitle'].str.contains('Java').sum()
data_count = df['jobtitle'].str.contains('Data').sum()
print('Data Jobs: {}, Java Jobs: {}'.format(data_count, java_count))