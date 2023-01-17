import numpy as np

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import os

print(os.listdir("../input"))
LiteracyRate = pd.read_csv("../input/LiteracyRate.csv")

LiteracyRate
LiteracyRate.head()
LiteracyRate.shape
LiteracyRate.columns
display(LiteracyRate['Age_Group'].unique())
col=['Age_Group', 'Urban_Male', 'Urban_Female', 'Urban_Total', 'Rural_Male',

       'Rural_Female', 'Rural_Total', 'Nepal_Male', 'Nepal_Female',

       'Nepal_Total']

df = pd.DataFrame(LiteracyRate, columns=col)
plt.figure(figsize=(6,5))

sns.barplot( df['Nepal_Total'],df['Age_Group'], alpha=0.8)

plt.xticks(rotation='horizontal')

plt.xlabel('Population', fontsize=14)

plt.ylabel('Age Group', fontsize=14)

plt.title("Literacy rate wrt age group in Nepal", fontsize=16)

plt.show()
plt.figure(figsize=(6,5))

sns.barplot( df['Nepal_Female'],df['Age_Group'], alpha=0.8)

plt.xticks(rotation='horizontal')

plt.xlabel('Population', fontsize=14)

plt.ylabel('Age Group', fontsize=14)

plt.title("Literacy rate wrt age group in Nepal", fontsize=16)

plt.show()
plt.figure(figsize=(6,5))

sns.barplot( df['Nepal_Male'],df['Age_Group'], alpha=0.5)

plt.xticks(rotation='horizontal')

plt.xlabel('Population', fontsize=14)

plt.ylabel('Age Group', fontsize=14)

plt.title("Literacy rate wrt age group in Nepal", fontsize=16)

plt.show()
plt.figure(figsize=(6,5))

sns.barplot( df['Rural_Total'],df['Age_Group'], alpha=0.8)

plt.xticks(rotation='horizontal')

plt.xlabel('Population', fontsize=14)

plt.ylabel('Age Group', fontsize=14)

plt.title("Literacy rate wrt age group in Nepal", fontsize=16)

plt.show()
plt.figure(figsize=(6,5))

sns.barplot( df['Rural_Female'],df['Age_Group'], alpha=0.8)

plt.xticks(rotation='horizontal')

plt.xlabel('Population', fontsize=14)

plt.ylabel('Age Group', fontsize=14)

plt.title("Literacy rate wrt age group in Nepal", fontsize=16)

plt.show()
plt.figure(figsize=(6,5))

sns.barplot( df['Rural_Male'],df['Age_Group'], alpha=0.8)

plt.xticks(rotation='horizontal')

plt.xlabel('Population', fontsize=14)

plt.ylabel('Age Group', fontsize=14)

plt.title("Literacy rate wrt age group in Nepal", fontsize=16)

plt.show()
plt.figure(figsize=(6,5))

sns.barplot( df['Urban_Total'],df['Age_Group'], alpha=0.8)

plt.xticks(rotation='horizontal')

plt.xlabel('Population', fontsize=14)

plt.ylabel('Age Group', fontsize=14)

plt.title("Literacy rate wrt age group in Nepal", fontsize=16)

plt.show()
plt.figure(figsize=(6,5))

sns.barplot( df['Urban_Female'],df['Age_Group'], alpha=0.8)

plt.xticks(rotation='horizontal')

plt.xlabel('Population', fontsize=14)

plt.ylabel('Age Group', fontsize=14)

plt.title("Literacy rate wrt age group in Nepal", fontsize=16)

plt.show()
plt.figure(figsize=(6,5))

sns.barplot( df['Urban_Male'],df['Age_Group'], alpha=0.8)

plt.xticks(rotation='horizontal')

plt.xlabel('Population', fontsize=14)

plt.ylabel('Age Group', fontsize=14)

plt.title("Literacy rate wrt age group in Nepal", fontsize=16)

plt.show()
plt.figure(figsize=(20,12))

for i in range(1,len(LiteracyRate)):

    plt.subplot(3,4,i)

    plt.title(df['Age_Group'][i])

    top = ['Urban Male','Urban Female']

    data = LiteracyRate.loc[df['Age_Group'] == df['Age_Group'][i],:]

    value =[float(data['Urban_Male']/data['Urban_Total'])*100,float(data['Urban_Female']/data['Urban_Total'])*100]

    plt.pie(value, labels=top, autopct='%1.1f%%',startangle=140)

    plt.axis('equal')

plt.show()
plt.figure(figsize=(20,12))

for i in range(1,len(LiteracyRate)):

    plt.subplot(3,4,i)

    plt.title(df['Age_Group'][i])

    top = ['Rural Male','Rural Female']

    data = LiteracyRate.loc[df['Age_Group'] == df['Age_Group'][i],:]

    value =[float(data['Rural_Male']/data['Rural_Total'])*100,float(data['Rural_Female']/data['Rural_Total'])*100]

    plt.pie(value, labels=top, autopct='%1.1f%%',startangle=140)

    plt.axis('equal')

plt.show()
plt.figure(figsize=(20,12))

for i in range(1,len(LiteracyRate)):

    plt.subplot(3,4,i)

    plt.title(df['Age_Group'][i])

    top = ['Total Male','Total Female']

    data = LiteracyRate.loc[df['Age_Group'] == df['Age_Group'][i],:]

    value =[float(data['Nepal_Male']/data['Nepal_Total'])*100,float(data['Nepal_Female']/data['Nepal_Total'])*100]

    plt.pie(value, labels=top, autopct='%1.1f%%',startangle=140)

    plt.axis('equal')

plt.show()