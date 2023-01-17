# importing libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#importing the data
df= pd.read_csv('../input/Admission_Predict.csv')
#getting the head of data
df.head()
df.shape
df.columns
# getting the data types of data
df.dtypes
## getting the mean,count etc of data
df.describe()

##bar plot
plt.subplots(figsize=(10,4))
sns.barplot(x="Research",y="Chance of Admit " ,data=df)
plt.subplots(figsize=(18,4))
sns.barplot(x="GRE Score",y="Chance of Admit ",data=df)
plt.subplots(figsize=(18,4))
sns.barplot(x="TOEFL Score",y="Chance of Admit ",data=df)
plt.subplots(figsize=(10,4))
sns.barplot(x="SOP",y="Chance of Admit ",data=df)
plt.subplots(figsize=(10,4))
sns.barplot(x="University Rating",y="Chance of Admit ",data=df)
plt.figure(figsize=(10,4))
sns.barplot(x= "LOR ",y="Chance of Admit ",data=df)
plt.subplots(figsize=(23,4))
sns.barplot(x="Chance of Admit ",y="CGPA",data=df)


#Pair plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df, hue="Chance of Admit ",
             vars=['CGPA','TOEFL Score','GRE Score'], size=5)
plt.show()
## Boxplots
plt.subplots (figsize=(18, 5))
sns.boxplot(x="GRE Score",y="Chance of Admit ",data=df)

plt.subplots (figsize=(15,5))
sns.boxplot(x="TOEFL Score",y="Chance of Admit ",data=df)

plt.subplots (figsize=(22,5))
sns.boxplot(x="Chance of Admit ",y="CGPA",data=df)

plt.subplots(figsize=(15,5))
sns.boxplot(x="LOR ",y="Chance of Admit ",data=df)

plt.subplots(figsize=(15,5))
sns.boxplot(x="SOP",y="Chance of Admit ",data=df)

plt.subplots(figsize=(15,5))
sns.boxplot(x="Research",y="Chance of Admit ",data=df)


            


#getting the correlation of data
fig,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()
