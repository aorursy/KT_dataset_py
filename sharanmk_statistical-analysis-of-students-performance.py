# Import necessary modules. 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
## Importing the dataset
stdPer = pd.read_csv("../input/StudentsPerformance.csv")
#Lets see the first five rows of the dataset
stdPer.head()
print ("The shape of data is (row, column):"+ str(stdPer.shape))
print (stdPer.info())
gender_list = stdPer['gender'].unique()
for gender in gender_list:
    print('Number of',gender,'students is',stdPer['gender'][stdPer['gender'] == gender].count())
    
#visualizing the above counts.
sns.countplot(x = 'gender', data = stdPer)
#Percentage distribution of gender count
# labels = ['female', 'male']

#using list comprehension here
sizes = [stdPer['gender'][stdPer['gender'] == gender].count() for gender in gender_list]
colors = ['lightcoral', 'lightskyblue']
 
# Plot
plt.pie(sizes, labels=gender_list, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140) 
plt.axis('equal')
plt.show()
#Actual count
race_ethnicity_list = stdPer['race/ethnicity'].unique()
for race_ethnicity in race_ethnicity_list:
    print('Number of students in',race_ethnicity,'is',stdPer['race/ethnicity'][stdPer['race/ethnicity'] == race_ethnicity].count())

#Plot using seaborn    
sns.countplot(y = 'race/ethnicity', data = stdPer, order = ['group C', 'group D', 'group B', 'group E', 'group A'])
#Percentage distribution of race/ethnicity count
labels = stdPer['race/ethnicity'].unique()
# labels = ['group B', 'group C', 'group A', 'group D', 'group E']

#using list comprehension here
sizes = [stdPer['race/ethnicity'][stdPer['race/ethnicity'] == 
        race_ethnicity].count() for race_ethnicity in labels]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','red']
explode = (0, 0.1, 0, 0,0)  # explode 1st slice
 
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
 
plt.axis('equal')
plt.show()
#Actual count
parent_edu_list = stdPer['parental level of education'].unique()
for parent_edu in parent_edu_list:
    print('Number of students in',parent_edu,'is',stdPer['race/ethnicity'][stdPer['parental level of education'] == parent_edu].count())

#Plot using seaborn    
sns.countplot(y = 'parental level of education', data = stdPer, order = ["some college","associate's degree", 'some high school', "bachelor's degree", "master's degree"])
#unique elements under parental level of eduction
labels = stdPer['parental level of education'].unique()

# Using list comprehension here
sizes = [stdPer['parental level of education'][stdPer['parental level of education'] == 
                                               qualification].count() for qualification in labels]

colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue','red',"blue"]

# Plot
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Percentage distribution of students count for different parental level of education')
plt.show()
preparation_list = stdPer['test preparation course'].unique()
for preparation in preparation_list:
    print('Number of students whose test preparation is',preparation, 'is',stdPer['test preparation course'][stdPer['test preparation course'] == preparation].count())
    
#visualizing the above counts.
sns.countplot(x = 'test preparation course', data = stdPer)
print('Avrage Maths Score : ',stdPer['math score'].mean())
print('Avrage Reading Score : ',stdPer['reading score'].mean())
print('Avrage Writing Score : ',stdPer['writing score'].mean())
fig, axes = plt.subplots(1,3,figsize=(15,5), sharey = True)
sns.despine()
sns.boxplot(y=stdPer['math score'], ax = axes[0], color = 'g')
sns.boxplot(y=stdPer['reading score'], ax = axes[1], color = 'b')
sns.boxplot(y=stdPer['writing score'], ax = axes[2], color = 'r')
plt.setp(axes, yticks=[i for i in range(0,110,10)])
plt.show()
#Above information can also be visualized by violin plots
f, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)
# sns.despine(left=True)

# Violinplot for distribution of math score
sns.violinplot(y = 'math score', data = stdPer, ax=axes[0], color = 'g')

# Violinplot for distribution of reading score
sns.violinplot(y = 'reading score', data = stdPer, ax=axes[1], color = 'b')

# Violinplot for distribution of writing score
sns.violinplot(y="writing score", data = stdPer, ax = axes[2], color = 'r')

# plt.setp(axes, yticks=[0,10,20,30,40,50,60,70,80,90,100])
plt.setp(axes, yticks=[i for i in range(0,110,10)])
plt.tight_layout()
# sns.set(style="white", palette="muted", color_codes=True)
fig, axes = plt.subplots(1,3, figsize = (10,5), sharey = True)
# sns.despine(left=True)

sns.distplot(stdPer['math score'], color="m", kde=False, ax=axes[0],bins=10, 
             hist_kws={"rwidth":0.9,'edgecolor':'black', 'alpha':1.0})

sns.distplot(stdPer['reading score'], color="g", kde=False, ax=axes[1],bins=10,
            hist_kws={"rwidth":0.9,'edgecolor':'black', 'alpha':1.0})

sns.distplot(stdPer['writing score'], color="b", kde=False, ax=axes[2],bins=10,
            hist_kws={"rwidth":0.9,'edgecolor':'black', 'alpha':1.0})

plt.tight_layout()

stdPer.groupby('gender').mean().reset_index()
f, axes = plt.subplots(1,3,figsize=(15, 5)) 
sns.boxplot(x = 'gender', y = 'math score', data = stdPer, ax=axes[0])
sns.boxplot(x = 'gender', y = 'reading score', data = stdPer, ax=axes[1])
sns.boxplot(x = 'gender', y = 'writing score', data = stdPer, ax=axes[2])
plt.setp(axes, yticks=[i for i in range(0,110,10)])
sns.despine()
plt.tight_layout()
stdPer.groupby('race/ethnicity').mean().reset_index().sort_values('math score', ascending = False)
f, axes = plt.subplots(3,1,figsize=(10, 15)) 
sns.boxplot(x = 'race/ethnicity', y = 'math score', data = stdPer, ax=axes[0])
sns.boxplot(x = 'race/ethnicity', y = 'reading score', data = stdPer, ax=axes[1])
sns.boxplot(x = 'race/ethnicity', y = 'writing score', data = stdPer, ax=axes[2])
plt.setp(axes, yticks=[i for i in range(0,110,10)])
sns.despine()
plt.tight_layout()
stdPer.groupby('parental level of education').mean().reset_index().sort_values('math score', ascending = False)
#We will plot three different boxplots
f, axes = plt.subplots(3,1,figsize=(10, 15)) 
sns.boxplot(x = 'parental level of education', y = 'math score', data = stdPer, ax=axes[0])
sns.boxplot(x = 'parental level of education', y = 'reading score', data = stdPer, ax=axes[1])
sns.boxplot(x = 'parental level of education', y = 'writing score', data = stdPer, ax=axes[2])
plt.setp(axes, yticks=[i for i in range(0,110,10)])
sns.despine()
plt.tight_layout()
f, axes = plt.subplots(1,3,figsize=(15, 5)) 
sns.violinplot(x = 'test preparation course', y = 'math score', data = stdPer, ax=axes[0])
sns.violinplot(x = 'test preparation course', y = 'reading score', data = stdPer, ax=axes[1])
sns.violinplot(x = 'test preparation course', y = 'writing score', data = stdPer, ax=axes[2])
plt.setp(axes, yticks=[i for i in range(0,110,10)])
sns.despine()
plt.show()
# plt.tight_layout()