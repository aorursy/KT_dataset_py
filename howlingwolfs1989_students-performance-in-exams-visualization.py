import numpy as np
import pandas as pd

#Matplot and Seabron Lib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
data = pd.read_csv('../input/StudentsPerformance.csv')
df = data.copy()
df.head()
df.info()
df.describe(include='all')
df.dtypes
df.isnull().sum()
sns.set(style="ticks", color_codes=True, font_scale=1)
plt.figure(figsize=(15, 15))

plt.subplot(321)
sns.countplot('gender', data=df);

plt.subplot(322)
sns.countplot('race/ethnicity', order=['group A', 'group B', 'group C', 'group D', 'group E'], data=df);

plt.subplot(323)
sns.countplot('test preparation course', data=df);

plt.subplot(324)
sns.countplot('lunch', data=df);

plt.figure(figsize=(33, 15))
plt.subplot(325)
sns.countplot('parental level of education', data=df);

sns.pairplot(df, hue="gender", height=3.5);
sns.pairplot(df, hue="race/ethnicity", height=3.5, hue_order=['group A', 'group B', 'group C', 'group D', 'group E']);
sns.pairplot(df, hue="lunch", height=3.5);
sns.pairplot(df, hue="test preparation course", height=3.5);
sns.pairplot(df, hue="parental level of education", height=3.5);
plt.figure(figsize=(18, 10))
plt.subplot(221)
sns.boxplot(x='math score', y='gender', data=df);

plt.subplot(222)
sns.boxplot(x='reading score', y='gender', data=df);

plt.subplot(223)
sns.boxplot(x='writing score', y='gender', data=df);
plt.figure(figsize=(18, 10))
plt.subplot(221)
sns.boxplot(x='math score', y='race/ethnicity', data=df, order=['group A', 'group B', 'group C', 'group D', 'group E']);

plt.subplot(222)
sns.boxplot(x='reading score', y='race/ethnicity', data=df, order=['group A', 'group B', 'group C', 'group D', 'group E']);

plt.subplot(223)
sns.boxplot(x='writing score', y='race/ethnicity', data=df, order=['group A', 'group B', 'group C', 'group D', 'group E']);
plt.figure(figsize=(30, 10))
plt.subplot(221)
sns.boxplot(x='math score', y='parental level of education', data=df);

plt.subplot(222)
sns.boxplot(x='reading score', y='parental level of education', data=df);

plt.subplot(223)
sns.boxplot(x='writing score', y='parental level of education', data=df);
plt.figure(figsize=(22, 10))
plt.subplot(221)
sns.boxplot(x='math score', y='lunch', data=df);

plt.subplot(222)
sns.boxplot(x='reading score', y='lunch', data=df);

plt.subplot(223)
sns.boxplot(x='writing score', y='lunch', data=df);
plt.figure(figsize=(22, 10))
plt.subplot(221)
sns.boxplot(x='math score', y='test preparation course', data=df);

plt.subplot(222)
sns.boxplot(x='reading score', y='test preparation course', data=df);

plt.subplot(223)
sns.boxplot(x='writing score', y='test preparation course', data=df);
sns.jointplot(x="math score", y="reading score", data=df, kind="kde");
sns.jointplot(x="math score", y="writing score", data=df, kind="kde");
sns.jointplot(x="reading score", y="writing score", data=df, kind="kde");
cor = df.corr()
sns.heatmap(cor, annot=True);