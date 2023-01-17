import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.set_style('darkgrid')
df = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
df.head()
df.info()
df.isnull().sum()
df['parental level of education'].unique()
#some high school is being considered as same as high school degree

#changing some college to college and some high school to high school

df['parental level of education'] = df['parental level of education'].replace({'some college':'college','some high school':'high school'})

df['parental level of education'].unique()
df.describe()
plt.figure(figsize = (10,15))



plt.subplot(3,1,1)

sns.distplot(df['math score'], kde = False, color='b')

plt.ylabel('Frequency')

plt.title('Math Score Distribution',fontsize=20)



plt.subplot(3,1,2)

sns.distplot(df['reading score'], kde = False, color='r')

plt.ylabel('Frequency')

plt.title('Reading Score Distribution',fontsize=20)



plt.subplot(3,1,3)

sns.distplot(df['writing score'], kde = False, color='g')

plt.ylabel('Frequency')

plt.title('Writing Score Distribution',fontsize=20)



plt.tight_layout()
# Creating columns of whether the students passed their class or not

# If the marks are < 40 means Fail otherwise Pass.



df["math grade"] = np.where(df["math score"]<40,"Fail" , "Pass")

df["reading grade"] = np.where(df["reading score"]<40, "Fail" , "Pass")

df["writing grade"] = np.where(df["writing score"]<40, "Fail" , "Pass")
df.head()
#Failed Students

failed_math = df['math grade'] == 'Fail'

failed_reading = df['reading grade'] == 'Fail'

failed_writing = df['writing grade'] == 'Fail'



fail_all= df[(failed_math) & (failed_reading) & (failed_writing)]

failed_students = df[(failed_math) | (failed_reading)|(failed_writing)]

passed_students = len(df) - len(failed_students)



print('Number of students who passed in all three subjects: ',passed_students)

print('Number of students who failed in atleast one subject: ',len(failed_students))

print('Number of students who failed in all three subjects: ',len(fail_all))
plt.figure(figsize=(5,5))



#Data to plot

labels = ['Passed Students', 'Failed Students']

color = ['green', 'red']

sizes = [passed_students,len(failed_students)]



#Plot

plt.pie(sizes,explode = (.25,0), labels = labels,autopct='%1.1f%%',shadow = True,colors=color)

plt.title('Percentage of Students who passed/failed in Exams',fontsize=15)

plt.show()
plt.figure(figsize=(5,5))



labels = df['gender'].unique()

size = df['gender'].value_counts()



plt.pie(size,labels=labels,autopct="%1.1f%%",shadow=True,explode=(0.05,0),startangle=90)

plt.title('Gender distribution',fontsize=15)

plt.show()
plt.figure(figsize=(15,5))

plt.rc('xtick',labelsize=15)

plt.rc('ytick',labelsize=12)



plt.subplot(1,3,1)

sns.boxplot(x = 'gender', y = 'math score', data = df, palette = "prism_r")

plt.title('Math Score Distribution',fontsize=20)



plt.subplot(1,3,2)

sns.boxplot(x = 'gender', y = 'reading score', data = df, palette = "prism_r")

plt.title('Reading Score Distribution',fontsize=20)



plt.subplot(1,3,3)

sns.boxplot(x = 'gender', y = 'writing score', data = df, palette = "prism_r")

plt.title('Writing Score Distribution',fontsize=20)



plt.tight_layout()
plt.figure(figsize=(15,5))





plt.subplot(1,3,1)

sns.countplot(x = "math grade" ,hue="gender" , data=df ,palette="winter_r")

plt.title("Math Grade")



plt.subplot(1,3,2)

sns.countplot(x = "reading grade" ,hue="gender" , data=df ,palette="winter_r")

plt.title("Reading Grade")



plt.subplot(1,3,3)

sns.countplot(x = "writing grade" ,hue="gender" , data=df ,palette="winter_r")

plt.title("Writing Grade")



plt.tight_layout()
plt.figure(figsize=(5,5))



df['race/ethnicity'].value_counts().plot.pie(autopct="%1.1f%%",shadow=True,explode=(0.05,0.05,0.05,0.05,0.05),startangle=90)

plt.title('Race/Ethinicity',fontsize=15)

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.barplot(x = "race/ethnicity" , y = "math score" , data = df, palette="magma")

plt.title("Math Scores",fontsize=20)



plt.subplot(1,3,2)

sns.barplot(x = "race/ethnicity" , y = "reading score" , data = df, palette="magma")

plt.title("Reading Scores",fontsize=20)



plt.subplot(1,3,3)

sns.barplot(x = "race/ethnicity" , y = "writing score" , data = df, palette="magma")

plt.title("Writing Scores",fontsize=20)



plt.tight_layout()
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.countplot(x = "math grade" ,hue="race/ethnicity" , data=df ,palette="viridis")

plt.title("Math Grade")



plt.subplot(1,3,2)

sns.countplot(x = "reading grade" ,hue="race/ethnicity" , data=df ,palette="viridis")

plt.title("Reading Grade")



plt.subplot(1,3,3)

sns.countplot(x = "writing grade" ,hue="race/ethnicity" , data=df ,palette="viridis")

plt.title("Writing Grade")



plt.tight_layout()
plt.figure(figsize=(6,6))



df['parental level of education'].value_counts().plot.pie(autopct="%1.1f%%",shadow=True,startangle=45,

                                                          explode = np.repeat(0.02,5))

plt.title('Parental level of Education',fontsize=20)



plt.show()
plt.figure(figsize=(20,7))



plt.subplot(1,3,1)

sns.barplot(x = "parental level of education" , y="math score" , data=df, palette="YlOrRd")

plt.xticks(rotation=90)

plt.title("Math Scores",fontsize=20)



plt.subplot(1,3,2)

sns.barplot(x = "parental level of education" , y="reading score" , data=df, palette="YlOrRd")

plt.xticks(rotation=90)

plt.title("Reading Scores",fontsize=20)



plt.subplot(1,3,3)

sns.barplot(x = "parental level of education" , y="writing score" , data=df, palette="YlOrRd")

plt.xticks(rotation=90)

plt.title("Writing Scores",fontsize=20)



plt.tight_layout()
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.countplot(x = "math grade" ,hue="parental level of education" , data=df ,palette="YlGn_r")

plt.title("Math Grade")



plt.subplot(1,3,2)

sns.countplot(x = "reading grade" ,hue="parental level of education" , data=df ,palette="YlGn_r")

plt.legend(loc="upper right")

plt.title("Reading Grade")



plt.subplot(1,3,3)

sns.countplot(x = "writing grade" ,hue="parental level of education" , data=df ,palette="YlGn_r")

plt.title("Writing Grade")



plt.tight_layout()
plt.figure(figsize=(5,5))



labels = df['test preparation course'].unique()

size = df['test preparation course'].value_counts()



plt.pie(size,labels=labels,autopct="%1.1f%%",shadow=True,explode=(0.04,0.04),startangle=90)

plt.title('Test Preparation Course',fontsize=15)

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.boxplot(x = "test preparation course" , y="math score" , data=df, palette='coolwarm')

plt.title("Math Scores",fontsize=20)



plt.subplot(1,3,2)

sns.boxplot(x = "test preparation course" , y="reading score" , data=df, palette='coolwarm')

plt.title("Reading Scores",fontsize=20)



plt.subplot(1,3,3)

sns.boxplot(x = "test preparation course" , y="writing score" , data=df, palette='coolwarm')

plt.title("Writing Scores",fontsize=20)



plt.tight_layout()
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.countplot(x = "math grade" ,hue="test preparation course" , data=df ,palette="inferno")

plt.title("Math Grade")



plt.subplot(1,3,2)

sns.countplot(x = "reading grade" ,hue="test preparation course" , data=df ,palette="inferno")

plt.title("Reading Grade")



plt.subplot(1,3,3)

sns.countplot(x = "writing grade" ,hue="test preparation course" , data=df ,palette="inferno")

plt.title("Writing Grade")



plt.tight_layout()
plt.figure(figsize=(5,5))



labels = df['lunch'].unique()

size = df['lunch'].value_counts()



plt.pie(size,labels=labels,autopct="%1.1f%%",shadow=True,explode=(0.04,0.04),startangle=90)

plt.title('lunch',fontsize=15)

plt.show()
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.boxplot(x = "lunch" , y="math score" , data=df, palette='summer')

plt.title("Math Scores",fontsize=20)



plt.subplot(1,3,2)

sns.boxplot(x = "lunch" , y="reading score" , data=df, palette='summer')

plt.title("Reading Scores",fontsize=20)



plt.subplot(1,3,3)

sns.boxplot(x = "lunch" , y="writing score" , data=df, palette='summer')

plt.title("Writing Scores",fontsize=20)



plt.tight_layout()
plt.figure(figsize=(15,5))



plt.subplot(1,3,1)

sns.countplot(x = "math grade" ,hue="lunch" , data=df ,palette="autumn")

plt.title("Math Grade")



plt.subplot(1,3,2)

sns.countplot(x = "reading grade" ,hue="lunch" , data=df ,palette="autumn")

plt.title("Reading Grade")



plt.subplot(1,3,3)

sns.countplot(x = "writing grade" ,hue="lunch" , data=df ,palette="autumn")

plt.title("Writing Grade")



plt.tight_layout()
plt.figure(figsize=(7,7))



df.groupby('lunch')['race/ethnicity'].value_counts().plot.pie(autopct="%1.1f%%",shadow=True,explode=np.repeat(0.02,10))

plt.title('Race/Ethinicity',fontsize=18)

plt.show()