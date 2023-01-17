import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

pass_mark = 60
df = pd.read_csv('../input/StudentsPerformance.csv')
df.columns = ['gender','race','parent_education','lunch','test_prep','math','reading','writing']
df.isna().sum()
df.head()
print('Male:',len(df[df.gender == 'male']))
print('Female:',len(df[df.gender == 'female']))
y = np.array([len(df[df.gender == 'male']),len(df[df.gender == 'female'])])
x = ["Male","Female"]
plt.bar(x,y)
plt.title("Gender Representation")
plt.xlabel("Gender")
plt.ylabel("Frequency")
plt.show()
x = ['Some college','Associates','Highschool','Some highschool','Bachelors','Masters']
y = df.parent_education.value_counts().values
plt.bar(x,y)
plt.title("Parent's Education")
plt.xlabel("Education Level")
plt.ylabel("Frequency")
plt.show()
no_college_idx = np.where([x in ['some college','high school','some high school'] for x in df.parent_education])[0]

df['College'] = 'Degree'
df.College[no_college_idx] = 'No Degree'
x = df.College.value_counts().index
y = df.College.value_counts().values

plt.bar(x,y)
plt.title("Parent's Degree")
plt.xlabel("College")
plt.ylabel("Frequency")
plt.show()
x = df.lunch.value_counts().index
y = df.lunch.value_counts().values
plt.bar(x,y)
plt.title("Lunch Payment")
plt.xlabel("Payment Type")
plt.ylabel("Frequency")
plt.show()
sns.pairplot(df)
fig,ax = plt.subplots(figsize=(6, 4))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()
plt.figure(figsize=(16,5))
plt.subplot(1, 3,1)
sns.boxplot(x="gender", y="math", data=df).set_title('Math')
plt.subplot(1, 3,2)
sns.boxplot(x="gender", y="reading", data=df).set_title('Reading')
plt.subplot(1, 3,3)
sns.boxplot(x="gender", y="writing", data=df).set_title('Writing')
plt.figure(figsize=(16,5))
plt.subplot(1, 3,1)
sns.boxplot(x="lunch", y="math", data=df).set_title('Math')
plt.subplot(1, 3,2)
sns.boxplot(x="lunch", y="reading", data=df).set_title('Reading')
plt.subplot(1, 3,3)
sns.boxplot(x="lunch", y="writing", data=df).set_title('Writing')
plt.figure(figsize=(16,5))
plt.subplot(1, 3,1)
sns.boxplot(x="College", y="math", data=df).set_title('Math')
plt.subplot(1, 3,2)
sns.boxplot(x="College", y="reading", data=df).set_title('Reading')
plt.subplot(1, 3,3)
sns.boxplot(x="College", y="writing", data=df).set_title('Writing')
failing = df[(df.math < pass_mark) & (df.writing < pass_mark) & (df.reading < pass_mark)]
y = np.array([len(failing[failing.gender == 'male']),len(failing[failing.gender == 'female'])])
x = ["Male","Female"]
plt.bar(x,y)
plt.title("Gender Representation")
plt.xlabel("Gender")
plt.ylabel("Frequency")
plt.show()
x = failing.lunch.value_counts().index
y = failing.lunch.value_counts().values

plt.bar(x,y)
plt.title("Lunch Payment")
plt.xlabel("Payment Type")
plt.ylabel("Frequency")
plt.show()
x = failing.College.value_counts().index
y = failing.College.value_counts().values

plt.bar(x,y)
plt.title("Parent's Degree")
plt.xlabel("College")
plt.ylabel("Frequency")
plt.show()
sns.pairplot(failing)
fig,ax = plt.subplots(figsize=(6, 4))
sns.heatmap(failing.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f',cmap="magma")
plt.show()