import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df=pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')
df.head()
df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
df.info()
fig, axs = plt.subplots(1, 4, figsize = (10, 10))
sns.boxplot(y = 'total_score', data = df, ax = axs[0])
sns.boxplot(y = 'math score', data = df, color = 'red', ax = axs[1])
sns.boxplot(y = 'writing score', data = df, color = 'green', ax = axs[2])
sns.boxplot(y = 'reading score', data = df, color = 'blue', ax = axs[3])
sns.pairplot(data = df, hue = 'gender', plot_kws={'alpha':0.5})
fig, axs = plt.subplots(4, 1, figsize=(10, 8))
sns.distplot(df['math score'], ax = axs[0])
sns.distplot(df['reading score'], ax = axs[1])
sns.distplot(df['writing score'], ax = axs[2])
sns.distplot(df['total_score'], ax = axs[3])
fig1, axs1 = plt.subplots(1, 4, figsize=(16, 6))
sns.barplot(y = 'total_score', x = 'test preparation course', data = df, ax = axs1[0])
sns.barplot(y = 'math score', x = 'test preparation course', data = df, ax = axs1[1])
sns.barplot(y = 'writing score', x = 'test preparation course', data = df, ax = axs1[2])
sns.barplot(y = 'reading score', x = 'test preparation course', data = df, ax = axs1[3])
fig2, axs2 = plt.subplots(1,4, figsize = (15, 5))
sns.barplot(y = 'total_score', x = 'race/ethnicity', data = df,hue = 'gender', ax = axs2[0] )
sns.barplot(y = 'math score', x = 'race/ethnicity', data = df, hue = 'gender', ax = axs2[1] )
sns.barplot(y = 'writing score', x = 'race/ethnicity', data = df, hue = 'gender', ax = axs2[2] )
sns.barplot(y = 'reading score', x = 'race/ethnicity', data = df, hue = 'gender', ax = axs2[3] )
plt.figure(figsize = (12, 6))
sns.countplot(x = 'parental level of education', data = df)
plt.figure(figsize = (12, 6))
sns.barplot(y = 'total_score', x = 'parental level of education', data = df)