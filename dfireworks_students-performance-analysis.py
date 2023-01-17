# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # visualization

%matplotlib inline

import seaborn as sns # visualization



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# some configs for seaborn

sns.set(style="whitegrid")

sns.set_palette("husl")
students=pd.read_csv("/kaggle/input/students-performance-in-exams/StudentsPerformance.csv")

students.rename(columns={"race/ethnicity": "ethnic",

                         "parental level of education": "parents_education",

                         "test preparation course":"test_prep_score",

                         "math score":"math",

                         "reading score":"reading",

                         "writing score":"writing"}, inplace=True)

students.info()
# is there Na observations?

students.isna().sum()
students.shape
students.columns
students.describe()
students["parents_education"].value_counts()
uniq_degree=students["parents_education"].unique()
plt.figure(dpi=100)

plt.pie(students["parents_education"].value_counts(),labels=uniq_degree,autopct="%1.2f%%")

plt.title("Parental level of education")
students["gender"].value_counts()
plt.figure(dpi=100)

plt.pie(students["gender"].value_counts(),labels=["Female","Male"],autopct="%1.2f%%")

plt.title("Students gender ratio")
students["ethnic"].unique()

plt.figure(dpi=100)

plt.pie(students["ethnic"].value_counts(),labels=["Group A","Group B","Group C","Group D","Group E"],autopct="%1.2f%%")

plt.title("Students gender ratio")
ax=sns.barplot(y="math",x="gender",data=students)

ax.set(xlabel="Gender(F/M)",ylabel="Math Score Mean",title="Mean of Math Scores by Gender")
print("Mean of math score (female)  = " + str(students[students["gender"]=="female"]["math"].mean()))

print("Mean of math score (male)    = " + str(students[students["gender"]=="male"]["math"].mean()))
ax=sns.barplot(x="ethnic",y="math",data=students)

ax.set(xlabel="Ethnicity",ylabel="Math Score Mean",title="Mean of Math Scores by Ethnicity")
print(students.groupby("ethnic")["math"].mean())
ax=sns.scatterplot(x="math",y="reading",data=students)

ax.set(xlabel="Math Scores",ylabel="Reading Scores",title="Math and Reading Scores")
ax = sns.regplot(x="math", y="reading", data=students,color="g",line_kws={'color':'blue'})

ax.set(xlabel="Math Scores",ylabel="Reading Scores",title="Math and Reading Scores with Regression Line")
ax=sns.scatterplot(x="math",y="writing",data=students)

ax.set(xlabel="Math Scores",ylabel="Writing Scores",title="Math and Writing Scores")
ax = sns.regplot(x="math", y="writing", data=students,color="g",line_kws={'color':'blue'})

ax.set(xlabel="Math Scores",ylabel="Reading Scores",title="Math and Writing Scores with Regression Line")
ax=sns.barplot(y="reading",x="gender",data=students)

ax.set(xlabel="Gender(F/M)",ylabel="Reding Score Mean",title="Mean of Reading Scores by Gender")
print("Mean of reading score (female)  = " + str(students[students["gender"]=="female"]["reading"].mean()))

print("Mean of reading score (male)    = " + str(students[students["gender"]=="male"]["reading"].mean()))
ax=sns.barplot(x="ethnic",y="reading",data=students)

ax.set(xlabel="Ethnicity",ylabel="Reading Score Mean",title="Mean of Reading Scores by Ethnicity")
print(students.groupby("ethnic")["reading"].mean())
ax=sns.scatterplot(x="reading",y="math",data=students,color="r")

ax.set(xlabel="Reading Scores",ylabel="Math Scores",title="Reading and Math Scores")
ax=sns.scatterplot(x="reading",y="writing",data=students,color="r")

ax.set(xlabel="Reading Scores",ylabel="Writing Scores",title="Reading and Writing Scores")
ax = sns.regplot(x="reading", y="writing", data=students,color="darkblue",line_kws={'color':'red'})

ax.set(xlabel="Reading Scores",ylabel="Writing Scores",title="Reading and Writing Scores with Regression Line")
ax=sns.barplot(y="writing",x="gender",data=students)

ax.set(xlabel="Gender(F/M)",ylabel="Writing Score Mean",title="Mean of Writing Scores by Gender")
print("Mean of writing score (female)  = " + str(students[students["gender"]=="female"]["writing"].mean()))

print("Mean of writing score (male)    = " + str(students[students["gender"]=="male"]["writing"].mean()))
ax=sns.barplot(x="ethnic",y="math",data=students)

ax.set(xlabel="Ethnicity",ylabel="Writing Score Mean",title="Mean of Writing Scores by Ethnicity")
print(students.groupby("ethnic")["writing"].mean())
ax=sns.scatterplot(x="writing",y="math",data=students,color="b")

ax.set(xlabel="Writing Scores",ylabel="Math Scores",title="Writing and Math Scores")
ax=sns.scatterplot(x="writing",y="reading",data=students,color="b")

ax.set(xlabel="Writing Scores",ylabel="Writing Scores",title="Writing and Reading Scores")
correlations=students.corr()

print(correlations)
plt.figure(dpi=100)

plt.title('Correlation Analysis of Math/Reading/Writing Scores')

sns.heatmap(correlations,annot=True,lw=1,linecolor='black',cmap='terrain')

plt.yticks(rotation=0)