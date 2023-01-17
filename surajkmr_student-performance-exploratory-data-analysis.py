import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
df_data = pd.read_csv("../input/StudentsPerformance.csv")

df_data.sample(5)
df_data.info()
df_data.describe()
# correlation between numerical data

sns.heatmap(df_data.corr(),annot=True)

plt.show()
df_data.isnull().sum()
df_data['race/ethnicity'].unique()
from matplotlib.pyplot import xticks

sns.countplot(x="race/ethnicity",  data = df_data)

xticks(rotation=90)

plt.xlabel('Race/Ethnicity')

plt.ylabel('Frequency')

plt.title('Race/Ethnicity ')
df_data["parental level of education"].unique()
from matplotlib.pyplot import xticks

sns.countplot(x="parental level of education",hue = "gender", data= df_data)

xticks(rotation=90)

plt.xlabel('levels of Education')

plt.ylabel('Frequency')

plt.title('Parental Level of Education')
df_data["lunch"].unique()
from matplotlib.pyplot import xticks

sns.countplot(x="lunch",hue = "gender", data= df_data)

xticks(rotation=90)

plt.xlabel('lunch types')

plt.ylabel('Frequency')

plt.title('lunch')
from matplotlib.pyplot import xticks

sns.countplot(x="lunch",hue = "parental level of education", data= df_data)

xticks(rotation=90)

plt.xlabel('lunch')

plt.ylabel('Frequency')

plt.title('lunch based on parental level of education')
from matplotlib.pyplot import xticks

sns.countplot(x="test preparation course",hue= "lunch", data= df_data)

xticks(rotation=90)

plt.xlabel('preparation course')

plt.ylabel('Frequency')
from matplotlib.pyplot import xticks

sns.countplot(x="test preparation course",hue= "parental level of education", data= df_data)

xticks(rotation=90)

plt.xlabel('preparation course')

plt.ylabel('Frequency')
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)

sns.distplot(df_data['math score'], kde = False, color='c', bins = 30)

plt.ylabel('Frequency')

plt.title('Math Score Distribution')



plt.subplot(1,3,2)

sns.distplot(df_data['reading score'], kde = False, color='c', bins = 30)

plt.ylabel('Frequency')

plt.title('reading Score Distribution')



plt.subplot(1,3,3)

sns.distplot(df_data['writing score'], kde = False, color='c', bins = 30)

plt.ylabel('Frequency')

plt.title('writing Score Distribution')
plt.figure(figsize=(15,4))

plt.subplot(1,3,1)

day_wise=df_data.groupby('gender')['math score'].mean().plot.bar()

plt.title("math score based on gender")

plt.ylabel('average math score', fontsize=12)

plt.xlabel('gender', fontsize=12)



plt.subplot(1,3,2)

day_wise=df_data.groupby('gender')['reading score'].mean().plot.bar()

plt.title("reading score based on gender")

plt.ylabel('average reading score', fontsize=12)

plt.xlabel('gender', fontsize=12)



plt.subplot(1,3,3)

day_wise=df_data.groupby('gender')['writing score'].mean().plot.bar()

plt.title("writing score based on gender")

plt.ylabel('average writing score', fontsize=12)

plt.xlabel('gender', fontsize=12)
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)

sns.boxplot(x=df_data['gender'],y=df_data['math score'])



plt.subplot(1,3,2)

sns.boxplot(x=df_data['gender'],y=df_data['reading score'])



plt.subplot(1,3,3)

sns.boxplot(x=df_data['gender'],y=df_data['writing score'])
plt.figure(figsize=(16,5))

plt.subplot(1,3,1)

day_wise=df_data.groupby('parental level of education')['math score'].mean().plot.bar()

plt.title("math score based on level of education of parents")

plt.ylabel('average math score', fontsize=12)

plt.xlabel('parents level of education', fontsize=12)



plt.subplot(1,3,2)

day_wise=df_data.groupby('parental level of education')['reading score'].mean().plot.bar()

plt.title("math score based on level of education of parents")

plt.ylabel('average math score', fontsize=12)

plt.xlabel('parents level of education', fontsize=12)



plt.subplot(1,3,3)

day_wise=df_data.groupby('parental level of education')['writing score'].mean().plot.bar()

plt.title("writing score based on level of education of parents")

plt.ylabel('average writing score', fontsize=12)

plt.xlabel('parents level of education', fontsize=12)
plt.figure(figsize=(16,5))

plt.subplot(1,3,1)

sns.stripplot(x="parental level of education",y='math score',data=df_data)

plt.xticks(rotation=90)



plt.subplot(1,3,2)

sns.stripplot(x="parental level of education",y='reading score',data=df_data)

plt.xticks(rotation=90)



plt.subplot(1,3,3)

sns.stripplot(x="parental level of education",y='writing score',data=df_data)

plt.xticks(rotation=90)
plt.figure(figsize=(16,5))

plt.subplot(1,3,1)

day_wise=df_data.groupby('race/ethnicity')['math score'].mean().plot.bar()

plt.title("math score based on race/ethnicity")

plt.ylabel('average math score', fontsize=12)

plt.xlabel('race/ethnicity', fontsize=12)



plt.subplot(1,3,2)

day_wise=df_data.groupby('race/ethnicity')['reading score'].mean().plot.bar()

plt.title("reading score based on race/ethnicity")

plt.ylabel('average reading score', fontsize=12)

plt.xlabel('race/ethnicity', fontsize=12)



plt.subplot(1,3,3)

day_wise=df_data.groupby('race/ethnicity')['writing score'].mean().plot.bar()

plt.title("writing score based on race/ethnicity")

plt.ylabel('average writing score', fontsize=12)

plt.xlabel('race/ethnicity', fontsize=12)
plt.figure(figsize=(16,5))

plt.subplot(1,3,1)

sns.stripplot(x="race/ethnicity",y='math score',data=df_data)

plt.xticks(rotation=90)



plt.subplot(1,3,2)

sns.stripplot(x="race/ethnicity",y='reading score',data=df_data)

plt.xticks(rotation=90)



plt.subplot(1,3,3)

sns.stripplot(x="race/ethnicity",y='writing score',data=df_data)

plt.xticks(rotation=90)
plt.figure(figsize=(16,5))

plt.subplot(1,3,1)

day_wise=df_data.groupby('test preparation course')['math score'].mean().plot.bar()

plt.title("math score based on test preparation course")

plt.ylabel('average math score', fontsize=12)

plt.xlabel('test preparation course', fontsize=12)



plt.subplot(1,3,2)

day_wise=df_data.groupby('test preparation course')['reading score'].mean().plot.bar()

plt.title("reading score based on test preparation course")

plt.ylabel('average reading score', fontsize=12)

plt.xlabel('test preparation course', fontsize=12)



plt.subplot(1,3,3)

day_wise=df_data.groupby('test preparation course')['writing score'].mean().plot.bar()

plt.title("writing score based on test preparation course")

plt.ylabel('average writing score', fontsize=12)

plt.xlabel('test preparation course', fontsize=12)
plt.figure(figsize=(16,5))

plt.subplot(1,3,1)

sns.boxplot(x=df_data['test preparation course'],y=df_data['math score'])



plt.subplot(1,3,2)

sns.boxplot(x=df_data['test preparation course'],y=df_data['reading score'])



plt.subplot(1,3,3)

sns.boxplot(x=df_data['test preparation course'],y=df_data['writing score'])
plt.figure(figsize=(16,5))

plt.subplot(1,3,1)

sns.boxplot(x=df_data['lunch'],y=df_data['math score'])



plt.subplot(1,3,2)

sns.boxplot(x=df_data['lunch'],y=df_data['reading score'])



plt.subplot(1,3,3)

sns.boxplot(x=df_data['lunch'],y=df_data['writing score'])
plt.figure(figsize=(15,5))

sns.lmplot(x="math score", y="reading score", hue="gender", data=df_data, markers=["o", "x"], palette="Set1")

plt.title('reading and math score with gender')



sns.lmplot(x="math score", y="writing score", hue="gender", data=df_data, markers=["o", "x"], palette="Set1")

plt.title('writing and  math score with gender')



sns.lmplot(x="reading score", y="writing score", hue="gender", data=df_data, markers=["o", "x"], palette="Set1")

plt.title('writing and reading score with gender')