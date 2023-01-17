# importing libraries



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df=pd.read_csv(r"../input/student-performance/StudentsPerformance.csv",header="infer")

# none is keyword so no double qoutes, but infer has



df.head(10)
df.columns
df['race/ethnicity'].unique()
df['parental level of education'].unique()
df['lunch'].unique()
df['test preparation course'].unique()
df.info()
df.dtypes
df.isnull().sum()
df.shape
df.describe()
df['gender'].value_counts()
df['race/ethnicity'].value_counts() #arranged in descending order of occurences
df['parental level of education'].value_counts()
df['test preparation course'].value_counts()
df['lunch'].value_counts()
plt.figure(figsize=(25,15))



plt.subplot(231)

plt.title("Gender",fontsize=30)

df['gender'].value_counts().plot.pie(autopct="%.2f%%",fontsize=15)



plt.subplot(232)

plt.title("Ethnicity",fontsize=30)

df['race/ethnicity'].value_counts().plot.pie(startangle=90,autopct="%.2f%%",fontsize=15)



plt.subplot(233)

plt.title("Parental Level Of Education",fontsize=30)

df['parental level of education'].value_counts().plot.pie(startangle=90,autopct="%.2f%%",fontsize=15)



plt.subplot(234)

plt.title("Lunch",fontsize=30)

df['lunch'].value_counts().plot.pie(autopct="%.2f",fontsize=15)



plt.subplot(235)

plt.title("Test Preparation Course",fontsize=30)

df['test preparation course'].value_counts().plot.pie(startangle=90,autopct="%.2f",fontsize=15)



plt.show()

plt.figure(figsize=(20,10))

sns.countplot(x='math score',data=df)

plt.xticks(fontsize=12)

plt.show()
plt.figure(figsize=(20,10))

sns.countplot(x='reading score',data=df,palette="muted")

plt.xticks(fontsize=13)

plt.show()
# palette colours: deep, muted, bright, pastel, dark, colorblind,Blues, husl(default), RdBu

# _d reverses order of colour



plt.figure(figsize=(20,10))

sns.countplot(x='writing score',data=df,palette="Blues_d")

plt.xticks(fontsize=13)

plt.show()
plt.figure(figsize=(20,15))

plt.subplots_adjust(hspace=0.35,wspace=0.25)



plt.subplot(221)

plt.title("Math score",fontsize=20)

sns.boxplot(x=df['math score'])



plt.subplot(222)

plt.title("Reading score",fontsize=20)

sns.boxplot(x=df['reading score'])



plt.subplot(223)

plt.title("Writing score",fontsize=20)

sns.boxplot(x=df['writing score'])



plt.show()
#plt.figure()

plt.subplots_adjust(top=2,bottom=0,right=4,left=1,wspace=0.5,hspace=0)

sns.set(font_scale=1.75)



plt.subplot(1,3,1)

sns.violinplot(y='math score',data=df)



plt.subplot(1,3,2)

sns.violinplot(y='reading score',data=df, color='g')



plt.subplot(1,3,3)

sns.violinplot(y='writing score',data=df,color='y')



plt.show()
# Meaure of central tendency of all three subjects

df.describe()
plt.figure(figsize=(15,6))

plt.subplots_adjust(top=1,bottom=0,right=1,left=0,hspace=0.25,wspace=0.5)



plt.subplot(131)

plt.title("Math score v/s Gender")

sns.barplot(x="gender",y='math score',data=df)



plt.subplot(132)

plt.title("Reading Score V/S Gender")

sns.barplot(x='gender',y='reading score',data=df)



plt.subplot(133)

plt.title("Writing Score V/S Gender")

sns.barplot(x='gender',y='writing score',data=df)



plt.show()
plt.figure(figsize=(15,15))

plt.subplots_adjust(left=0.125,bottom=0.1, right=0.9,  top=0.9, wspace=0.35,hspace=0.5)

#hspace is space b/w rows, wspace=space b/w columns



plt.subplot(221)

plt.title("Math Score v/s Ethnicity")

sns.barplot(x='race/ethnicity',y='math score',data=df)

plt.xticks(rotation=90)



plt.subplot(222)

plt.title("Reading Score v/s Ethnicity")

sns.barplot(x='race/ethnicity',y='reading score',data=df)

plt.xticks(rotation='vertical')



plt.subplot(223)

plt.title("Writing Score v/s Ethnicity")

sns.barplot(y='race/ethnicity',x='writing score',data=df)



plt.show()
plt.figure(figsize=(15,6))

plt.subplots_adjust(wspace=0.5)



plt.subplot(1,3,1)

plt.title("Math score v/s Preparation course",fontsize=15)

sns.barplot(x='test preparation course',y='math score',data=df)



plt.xlabel("test preparation course",labelpad=10)

#to increase space between axes and axes label



plt.subplot(1,3,2)

plt.title("Reading score v/s Preparation course",fontsize=15)

sns.barplot(x='test preparation course',y='reading score',data=df)

plt.xlabel("test preparation course",labelpad=10)



plt.subplot(1,3,3)

plt.title("Writing score v/s Preparation course",fontsize=15)

g=sns.barplot(x='test preparation course',y='writing score',data=df)

plt.xlabel("test preparation course",labelpad=10)



plt.show()
plt.figure(figsize=(15,15))

plt.subplots_adjust(hspace=1,wspace=0.5)



plt.subplot(2,2,1)

plt.title("Math score v/s Parental level of education ")

sns.barplot(x='parental level of education',y="math score",data=df)

plt.xticks(rotation=90)



plt.subplot(2,2,2)

plt.title("Reading score v/s Parental level of education")

sns.barplot(x='parental level of education',y="reading score",data=df)

plt.xticks(rotation=90)



plt.subplot(2,2,3)

plt.title("Writing score v/s Parental level of education")

sns.barplot(x='parental level of education',y="writing score",data=df)

plt.xticks(rotation=90)



plt.show()
plt.figure(figsize=(20,10))

plt.subplots_adjust(wspace=0.3)



plt.subplot(1,3,1)

plt.title("Math Score v/s Lunch")

sns.barplot(x='lunch',y="math score",data=df)



plt.subplot(1,3,2)

plt.title("Reading Score v/s Lunch")

sns.barplot(x='lunch',y="reading score",data=df)



plt.subplot(1,3,3)

plt.title("Writing Score v/s Lunch")

sns.barplot(x='lunch',y="writing score",data=df)



plt.show()

plt.figure(figsize=(17,12))

plt.subplots_adjust(wspace=0.25,hspace=0.8)



plt.subplot(221)

plt.title("Math score",fontsize=25)

sns.barplot(x='race/ethnicity',y='math score',hue='test preparation course',data=df)

plt.xticks(rotation=90)



plt.subplot(222)

plt.title("Reading score",fontsize=25)

sns.barplot(x='race/ethnicity',y='reading score',hue='test preparation course',data=df)

plt.xticks(rotation=90)



plt.subplot(223)

plt.title("Writing score",fontsize=25)

sns.barplot(x='race/ethnicity',y='writing score',hue='test preparation course',data=df)

plt.xticks(rotation=90)



plt.show()
plt.figure(figsize=(30,40))

plt.subplots_adjust(hspace=0.4)



plt.subplot(311)

plt.title("Math score",fontsize=40)

sns.barplot(x='race/ethnicity',y='math score',hue='parental level of education',data=df)

plt.xticks(rotation=90)

plt.legend(loc='lower right')



plt.subplot(312)

plt.title("Reading score",fontsize=40)

sns.barplot(x='race/ethnicity',y='reading score',hue='parental level of education',data=df)

plt.xticks(rotation=90)

plt.legend(loc='lower right')



plt.subplot(313)

plt.title("Writing score",fontsize=40)

sns.barplot(x='race/ethnicity',y='writing score',hue='parental level of education',data=df)

plt.xticks(rotation=90)

plt.legend(loc='lower right')



plt.show()
plt.figure(figsize=(20,10))

plt.subplots_adjust(wspace=0.25)



plt.subplot(131)

plt.title("Math Score")

sns.barplot(x="gender",y="math score",hue='test preparation course',data=df)



plt.subplot(132)

plt.title("Reading Score")

sns.barplot(x="gender",y="reading score",hue='test preparation course',data=df)



plt.subplot(133)

plt.title("Writing Score")

sns.barplot(x="gender",y="writing score",hue='test preparation course',data=df)



plt.show()
plt.figure(figsize=(20,12))

plt.subplots_adjust(wspace=0.25,hspace=0.5)



plt.subplot(221)

plt.title("Math Score")

sns.barplot(x="gender",y="math score",hue='race/ethnicity',data=df)



plt.subplot(222)

plt.title("Reading Score")

sns.barplot(x="gender",y="reading score",hue='race/ethnicity',data=df)



plt.subplot(223)

plt.title("Writing Score")

sns.barplot(x="gender",y="writing score",hue='race/ethnicity',data=df)



plt.show()
plt.figure(figsize=(20,17))

plt.subplots_adjust(wspace=0.25,hspace=0.9)



plt.subplot(221)

plt.title("Math Score")

sns.barplot(x='parental level of education',y="math score",hue='test preparation course',data=df)

plt.xticks(rotation=90)



plt.subplot(222)

plt.title("Reading Score")

sns.barplot(x='parental level of education',y="reading score",hue='test preparation course',data=df)

plt.xticks(rotation=90)



plt.subplot(223)

plt.title("Writing Score")

sns.barplot(x='parental level of education',y="writing score",hue='test preparation course',data=df)

plt.xticks(rotation=90)



plt.show()
plt.figure(figsize=(20,15))

plt.subplots_adjust(wspace=0.35,hspace=0.5)



plt.subplot(2,2,1)

plt.title("Ethnicity v/s Preparaton Course")

sns.countplot(x='race/ethnicity',hue='test preparation course',data=df)



plt.subplot(2,2,2)

plt.title("Gender v/s Preparaton Course")

sns.countplot(x="gender",hue='test preparation course',data=df)



plt.subplot(2,2,3)

plt.title("Parent's degree v/s Preparaton Course")

sns.countplot(x="parental level of education",hue='test preparation course',data=df)

plt.xticks(rotation=90)



plt.subplot(2,2,4)

plt.title("Lunch v/s Preparaton Course")

sns.countplot(x='lunch',hue='test preparation course',data=df)



plt.show()
plt.figure(figsize=(20,10))

plt.subplots_adjust(wspace=0.25)



plt.subplot(1,2,1)

plt.title("Parent's degree v/s Lunch")

sns.countplot(x="parental level of education",hue='lunch',data=df)

plt.xticks(rotation=90)



plt.subplot(1,2,2)

plt.title("Parent's degree v/s Gender")

sns.countplot(x="parental level of education",hue='gender',data=df)

plt.xticks(rotation=90)



plt.show()
plt.figure(figsize=(20,15))

plt.subplots_adjust(hspace=0.35, wspace=0.25)



plt.subplot(221)

plt.title("Ethnicity v/s Lunch")

sns.countplot(x='race/ethnicity', hue='lunch',data=df)



plt.subplot(222)

plt.title("Ethnicity v/s Gender")

sns.countplot(x='race/ethnicity', hue='gender',data=df)



plt.subplot(223)

plt.title("Ethnicity v/s Parent's Education")

sns.countplot(x='race/ethnicity', hue="parental level of education",data=df)

plt.legend(loc="upper left")



plt.show()
plt.figure(figsize=(15,6))

plt.title("Gender v/s Lunch",fontsize=30)

sns.countplot(x="gender", hue="lunch",data=df)

plt.show()
df.corr()
sns.heatmap(df.corr())
data=pd.read_csv(r"../input/student-performance/StudentsPerformance.csv",header="infer")

data.head()
# label encoding: converting categorical to numerical values

from sklearn import preprocessing

label_encoder=preprocessing.LabelEncoder()



data['gender']=label_encoder.fit_transform(data['gender'])

data['race/ethnicity']=label_encoder.fit_transform(data['race/ethnicity'])

data['parental level of education']=label_encoder.fit_transform(data['parental level of education'])

data['lunch']=label_encoder.fit_transform(data['lunch'])

data['test preparation course']=label_encoder.fit_transform(data['test preparation course'])



data.head()

data.corr()

#who did well in reading did well in writing--> heatmap
sns.heatmap(data.corr())