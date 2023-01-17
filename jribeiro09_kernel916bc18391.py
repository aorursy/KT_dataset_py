import pandas as pd 

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
df = pd.read_csv("../input/StudentsPerformance.csv")
df.head()
df.info(0)
df = df.rename(columns={'race/ethnicity':'ethnicity', 'parental level of education' : 'parents_education', 'test preparation course':'test_preparation_course', 'math score' : 'math_score', 'reading score' : 'reading_score', 'writing score' : 'writing_score' })
df.head()
df.isnull().sum()
df.ethnicity.value_counts()
df.parents_education.value_counts()
df.lunch.value_counts()
df.test_preparation_course.value_counts()
df.describe()
df['mean_score'] = df.mean(axis=1)
# Criando a função para a converção

def ScoretoGrade(mscore):    

    if (mscore >= 90 ):

        return 'A'

    if (mscore >= 80):

        return 'B'

    if (mscore >= 70):

        return 'C'

    if (mscore >= 60):

        return 'D'

    else: 

        return 'E/F'



# Criando a coluna com as notas novas    

df['grade'] = df.apply(lambda x : ScoretoGrade(x['mean_score']), axis=1)

            
df.head()
# Alunos bons

Top_students = df[df.grade == 'A']

# Alunos ruins

Fail_students = df[df.grade == 'E/F']
Top_students.head()
Fail_students.head()
mean_math = df['math_score'].mean()

mean_reading = df['reading_score'].mean()

mean_writing = df['writing_score'].mean()
# plotando os gráficos

plt.hist(df['math_score'], rwidth=0.9, edgecolor='k')

# Adicionando as legendas

plt.xlabel('Score')

plt.ylabel('Frequency')

# Adicionando o titulo

plt.title('Histrogram Math Score')

# Adicionando a linha de média

plt.axvline(mean_math, color = 'k', linestyle='dashed', linewidth=3)

# Adicionando a linha de nota minima

plt.axvline(60, color = 'r', linestyle='dashed', linewidth=3)

# Adicionando legendas

plt.legend(('mean','scores'))

plt.show()
plt.hist(df['reading_score'], rwidth=0.9, edgecolor='k')

plt.xlabel('Score')

plt.ylabel('Frequency')

plt.title('Histrogram Reading Score')

plt.axvline(mean_reading, color = 'k', linestyle='dashed', linewidth=2.5)

plt.axvline(60, color = 'r', linestyle='dashed', linewidth=3)

plt.legend(('Mean','Acceptable Score'))

plt.show()
plt.hist(df['writing_score'], rwidth=0.9, edgecolor='k')

plt.xlabel('Score')

plt.ylabel('Frequency')

plt.title('Histrogram Writing Score')

plt.axvline(mean_writing, color = 'k', linestyle='dashed', linewidth=3)

plt.axvline(60, color = 'r', linestyle='dashed', linewidth=3)

plt.legend(('Mean','Acceptable Score'))

plt.show()
df.grade.value_counts()
sns.countplot(x="grade", data = df, order=['A','B','C','D','E/F'],  palette="muted")

plt.title('Grade count')

plt.show()
sns.countplot(x='ethnicity', data = Top_students, palette="muted")

plt.title('Ethnicity of Top students')

plt.show()
plt.pie(Top_students.test_preparation_course.value_counts(), labels=['none','completed'], autopct='%1.1f%%', colors = ['magenta', 'cyan'])

my_circle=plt.Circle( (0,0), 0.75, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.axis('equal')

plt.title('Preparation of Top students')

plt.show()
plt.pie(Top_students.lunch.value_counts(), labels=['standard','free/reduce'], autopct='%1.1f%%')

my_circle=plt.Circle( (0,0), 0.8, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.axis('equal')

plt.title('Lunch of Top Students')

plt.show()
p = sns.countplot(x='parents_education', data = Top_students, palette="muted")

plt.setp(p.get_xticklabels(), rotation=45)

plt.title('Parents Education of Top students')

plt.show()
sns.countplot(x='ethnicity', data = Fail_students, palette="muted")

plt.title('Ethnicity of students with low grade')

plt.show()
plt.pie(Fail_students.test_preparation_course.value_counts(), labels=['none','completed'], autopct='%1.1f%%', labeldistance = 1.1,colors = ['magenta', 'cyan'])

my_circle=plt.Circle( (0,0), 0.79, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.axis('equal')

plt.title('Preparation of students with low grade')

plt.show()
plt.pie(Fail_students.lunch.value_counts(), labels=['standard','free/reduce'], autopct='%1.1f%%')

my_circle=plt.Circle( (0,0), 0.75, color='white')

p=plt.gcf()

p.gca().add_artist(my_circle)

plt.axis('equal')

plt.title('Lunch of students with low grade')

plt.show()
p = sns.countplot(x='parents_education', data = Fail_students, palette="muted")

plt.setp(p.get_xticklabels(), rotation=45)

plt.title('Parents Education of students with low grade')

plt.show()
df_f = df[df.gender == 'female']

df_m = df[df.gender == 'male']
df_f.gender[df_f.gender == 'male'].count()
df_m.gender[df_m.gender == 'female'].count()
sns.distplot(df_m['math_score'])

sns.distplot(df_f['math_score'])

plt.title('Histogram of math score by gender')

plt.legend(('Male','Female'))

plt.show()
sns.distplot(df_m['reading_score'])

sns.distplot(df_f['reading_score'])

plt.title('Histogram of reading score by gender')

plt.legend(('Male','Female'))

plt.show()
sns.distplot(df_m['writing_score'])

sns.distplot(df_f['writing_score'])

plt.title('Histogram of writing score by gender')

plt.legend(('Male','Female'))

plt.show()
sns.distplot(df_m['mean_score'])

sns.distplot(df_f['mean_score'])

plt.title('Histogram of mean score by gender')

plt.legend(('Male','Female'))

plt.show()