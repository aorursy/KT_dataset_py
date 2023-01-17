import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data= pd.read_csv('../input/StudentsPerformance.csv')
data.shape
data.head()
data.isnull().sum()

data['total_score']= data['math score']+ data['reading score']+ data['writing score']
data.head()
data['gender'].unique()
sns.set(style="ticks", color_codes=True)
sns.catplot(x="gender", kind='count' , palette= "ch:.25", data=data);
data['race/ethnicity'].unique()
sns.set(style="ticks", color_codes=True)
sns.catplot(x="race/ethnicity", kind='count' , palette= "ch:.25", data=data);
data['parental level of education'].unique()
sns.set(style="ticks", color_codes=True)
sns.catplot(x="parental level of education", kind='count' , palette= "pastel", data=data);
data['lunch'].unique()
data['test preparation course'].unique()
data.groupby('gender', axis = 0)['lunch'].count()
sns.set(style="ticks", color_codes=True)
sns.catplot(x="gender", y="math score", kind='swarm' ,data=data);
sns.set(style="ticks", color_codes=True)
sns.catplot(x="gender", y="reading score", kind='swarm' ,data=data);
sns.set(style="ticks", color_codes=True)
sns.catplot(x="gender", y="writing score", kind='swarm' ,data=data);
sns.set(style="ticks", color_codes=True)
sns.catplot(x="gender", y="total_score", kind='swarm' ,data=data);