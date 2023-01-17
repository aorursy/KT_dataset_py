#Import the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#read the csv file
data=pd.read_csv('../input/StudentsPerformance.csv')
data.shape
data.head()
data.describe()
data.info()
#check for any missing values
data.isnull().sum()
#Create total_score variable by adding 3 different subjects
data['total_score'] = data['math score'] + data['reading score'] + data['writing score']
#Create percentage variable from total_score
data['percentage'] = data['total_score']/3
#Create results variable with minimum score
data['result']=[0 if data['math score'][x]<36 or data['reading score'][x]<36 or data['writing score'][x]<36 else 1 for x in range(len(data))]
data.head()
def getCount(code):
    i=0
    if code == 'a_80':
        for c in data['percentage']:
            if c>=80:
                i+=1
    elif code == 'b_70_80':
        for c in data['percentage']:
            if c>=70 and c<80:
                i+=1
    elif code == 'b_60_70':
        for c in data['percentage']:
            if c>=60 and c<70:
                i+=1
    elif code == 'b_60':
        for c in data['percentage']:
            if c<60:
                i+=1
    return i

a_80 = getCount('a_80')/1000
b_70_80 = getCount('b_70_80')/1000
b_60_70 = getCount('b_60_70')/1000
b_60 = getCount('b_60')/1000

print(a_80*100)
print(b_70_80*100)
print(b_60_70*100)
print(b_60*100)
labels = ['Above 80%', 'Between 70%-80%', 'Between 60%-70%', 'Below 60%']
sizes = [a_80*100, b_70_80*100, b_60_70*100, b_60*100]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(12,5))
sns.countplot(x='gender', data = data, hue='result', palette='bright')
fig, ax = plt.subplots(figsize=(12,5))
sns.countplot(x='race/ethnicity', data = data, hue='result', palette='bright')
fig, ax = plt.subplots(figsize=(12,5))
sns.countplot(x='parental level of education', data = data, hue='result', palette='bright')
fig, ax = plt.subplots(figsize=(12,5))
sns.countplot(x='test preparation course', data = data, hue='result', palette='bright')
fig, ax = plt.subplots(figsize=(12,5))
sns.countplot(x='lunch', data = data, hue='result', palette='bright')
corr = data.corr()
corr
sns.heatmap(data.corr(),annot=True,linewidths=1,fmt=".2f")
plt.show()
sns.pairplot(data)
table = data.sort_values(by=['total_score'],ascending=False)
table
table = pd.pivot_table(data,index = ['gender','test preparation course','race/ethnicity','parental level of education'])
table