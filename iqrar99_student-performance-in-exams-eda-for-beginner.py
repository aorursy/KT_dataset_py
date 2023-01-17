#importing all important packages

import numpy as np #linear algebra

import pandas as pd #data processing

import matplotlib.pyplot as plt #data visualisation

import seaborn as sns #data visualisation

%matplotlib inline
#input data

data = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

data.head(10)
data.info() #checking data type for each column
data.shape
data.describe() #starter code to find out some basic statistical insights
print(data.isna().sum())
data['gender'].value_counts()
data.iloc[:,1].value_counts()
data.iloc[:,2].value_counts()
data.iloc[:,3].value_counts()
data.iloc[:,4].value_counts()
male = data[data['gender'] == 'male']

female = data[data['gender'] != 'male']



print("Math Score")

print("Male    :",round(male['math score'].sum()/len(male),3))

print("Female  :",round(female['math score'].sum()/len(female),3),'\n')



print("Reading Score")

print("Male    :",round(male['reading score'].sum()/len(male),3))

print("Female  :",round(female['reading score'].sum()/len(female),3),'\n')



print("Writing Score")

print("Male    :",round(male['writing score'].sum()/len(male),3))

print("Female  :",round(female['writing score'].sum()/len(female),3))
scores = pd.DataFrame(data['math score'] + data['reading score'] + data['writing score'], columns = ["total score"])

scores = pd.merge(data,scores, left_index = True, right_index = True).sort_values(by=['total score'],ascending=False)

scores.head(30)
sns.set(style="darkgrid")

f, axs = plt.subplots(1,2, figsize = (20,8))



sns.countplot(x = 'race/ethnicity', data = data, ax = axs[0]) #race / ethnicity

sns.countplot(x = 'parental level of education', data = data, ax = axs[1]) #parental level of education



plt.xticks(rotation=90)



plt.show()
f, ax = plt.subplots(1,1, figsize = (15,5))



sns.countplot(x = 'race/ethnicity', data = data, hue = 'gender', palette = 'Set1') #race / ethnicity



plt.show()
f, axs = plt.subplots(2,1, figsize = (15,10))



sns.countplot(x = 'parental level of education', data = data, hue = 'lunch', ax = axs[0], palette = 'spring')

sns.countplot(x = 'race/ethnicity', data = data, hue = 'lunch', ax = axs[1] , palette = 'spring')



plt.show()
sns.set_style('whitegrid')



f, ax = plt.subplots(1,1,figsize= (15,5))

ax = sns.swarmplot(x = 'math score', y='gender', data = data, palette = 'Set1') #math score

plt.show()



f, ax = plt.subplots(1,1,figsize= (15,5))

ax = sns.swarmplot(x = 'reading score', y='gender', data = data, palette = 'Set1') #reading score

plt.show()



f, ax = plt.subplots(1,1,figsize= (15,5))

ax = sns.swarmplot(x = 'writing score', y='gender', data = data, palette = 'Set1') #writing score

plt.show()



sns.jointplot(x ='math score', y = 'writing score', data = data, color = 'green', height = 8, kind = 'reg')

sns.jointplot(x ='math score', y = 'writing score', data = data, color = 'green', height = 8, kind = 'hex')
sns.jointplot(x ='math score', y = 'reading score', data = data, color = 'blue', height = 8, kind = 'reg')

sns.jointplot(x ='math score', y = 'reading score', data = data, color = 'blue', height = 8, kind = 'hex')
sns.jointplot(x ='reading score', y = 'writing score', data = data, color = 'red', height = 8, kind = 'reg')

sns.jointplot(x ='reading score', y = 'writing score', data = data, color = 'red', height = 8, kind = 'hex')
#math score

passed = len(data[data['math score'] >= 60])

not_passed = 1000 - passed



percentage1 = [passed, not_passed]



#reading score

passed = len(data[data['reading score'] >= 60])

not_passed = 1000 - passed



percentage2 = [passed, not_passed]



#writing score

passed = len(data[data['writing score'] >= 60])

not_passed = 1000 - passed



percentage3 = [passed, not_passed]
labels = "Passed", "Not Passed"



f, axs = plt.subplots(2,2, figsize=(15,10))



#Math Score

axs[0,0].pie(percentage1, labels = labels, explode=(0.05,0.05), autopct = '%1.1f%%', startangle = 200, colors = ["#e056fd", "#badc58"])

axs[0,0].set_title("Math Score", size = 20)

axs[0,0].axis('equal')



#Reading Score

axs[0,1].pie(percentage2, labels = labels, explode=(0.05,0.05), autopct = '%1.1f%%', startangle = 200, colors = ["#e056fd", "#badc58"])

axs[0,1].set_title("Reading Score", size = 20)

axs[0,1].axis('equal')



#Writing Score

axs[1,0].pie(percentage3, labels = labels, explode=(0.05,0.05), autopct = '%1.1f%%', startangle = 200, colors = ["#e056fd", "#badc58"])

axs[1,0].set_title("Writing Score", size = 20)

axs[1,0].axis('equal')



f.delaxes(axs[1,1]) #deleting axs[1,1] so it will be white blank



plt.show()
f, ax = plt.subplots(1,1,figsize=(10,5))



plt.hist(data['math score'], 30, color = 'skyblue')

ax.set(title = "Math Score")



plt.show()
f, ax = plt.subplots(1,1,figsize=(10,5))



plt.hist(data['reading score'], 30, color = 'salmon')

ax.set(title = "Reading Score")



plt.show()
f, ax = plt.subplots(1,1,figsize=(10,5))



plt.hist(data['writing score'], 30, color = 'goldenrod') #visit https://xkcd.com/color/rgb/ if you want to see more colors

ax.set(title = "Writing Score")



plt.show()
sns.heatmap(data.corr(), annot= True, cmap = 'autumn') #data.corr() used to make correlation matrix



plt.show()