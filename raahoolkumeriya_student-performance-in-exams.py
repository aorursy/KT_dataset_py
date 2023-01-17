import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
passmark = 35
dataset = pd.read_csv("../input/StudentsPerformance.csv")
dataset.head()
print (dataset.shape, dataset.size, len(dataset))
dataset.describe()
dataset.isnull().sum()
dataset.info()
dir(sns)
plt.figure(figsize=(25,16))

p = sns.countplot(x="math score", data = dataset, palette="muted")

_ = plt.setp(p.get_xticklabels(), rotation=90) 

dataset['Math_PassStatus'] = np.where(dataset['math score']<passmark, 'Fail', 'Pass')

dataset.Math_PassStatus.value_counts()
plt.figure(figsize=(20,10))

p = sns.countplot(x='parental level of education', data = dataset, hue='Math_PassStatus', palette='bright')

_ = plt.setp(p.get_xticklabels(), rotation=90) 
plt.figure(figsize=(25,16))

sns.countplot(x="reading score", data = dataset, palette="muted")

plt.show()
dataset['Reading_PassStatus'] = np.where(dataset['reading score']<passmark, 'Fail', 'Pass')

dataset.Reading_PassStatus.value_counts()
plt.figure(figsize=(20,10))

p = sns.countplot(x='parental level of education', data = dataset, hue='Reading_PassStatus', palette='bright')

_ = plt.setp(p.get_xticklabels(), rotation=90) 
plt.figure(figsize=(25,16))

p = sns.countplot(x="writing score", data = dataset, palette="muted")

_ = plt.setp(p.get_xticklabels(), rotation=90) 
dataset['Writing_PassStatus'] = np.where(dataset['writing score']<passmark, 'Fail', 'Pass')

dataset.Writing_PassStatus.value_counts()
plt.figure(figsize=(20,10))

p = sns.countplot(x='parental level of education', data = dataset, hue='Writing_PassStatus', palette='bright')

_ = plt.setp(p.get_xticklabels(), rotation=90) 
dataset['OverAll_PassStatus'] = dataset.apply(lambda x : 'Fail' if x['Math_PassStatus'] == 'Fail' or

                                    x['Reading_PassStatus'] == 'Fail' or x['Writing_PassStatus'] == 'Fail' else 'Pass', axis =1)



dataset.OverAll_PassStatus.value_counts()
plt.figure(figsize=(20,10))

p = sns.countplot(x='parental level of education', data = dataset, hue='OverAll_PassStatus', palette='bright')

_ = plt.setp(p.get_xticklabels(), rotation=90) 
dataset['Total_Marks'] = dataset['math score']+dataset['reading score']+dataset['writing score']

dataset['Percentage'] = dataset['Total_Marks']/3
plt.figure(figsize=(25,16))

p = sns.countplot(x="Percentage", data = dataset, palette="muted")

_ = plt.setp(p.get_xticklabels(), rotation=0) 
def GetGrade(Percentage, OverAll_PassStatus):

    if ( OverAll_PassStatus == 'Fail'):

        return 'Fail'    

    if ( Percentage >= 80 ):

        return 'A'

    if ( Percentage >= 70):

        return 'B'

    if ( Percentage >= 60):

        return 'C'

    if ( Percentage >= 50):

        return 'D'

    if ( Percentage >= 40):

        return 'E'

    else: 

        return 'Fail'



dataset['Grade'] = dataset.apply(lambda x : GetGrade(x['Percentage'], x['OverAll_PassStatus']), axis=1)



dataset.Grade.value_counts()
plt.figure(figsize=(20,10))

sns.countplot(x="Grade", data = dataset, order=['A','B','C','D','E','Fail'],  palette="muted")

plt.show()
plt.figure(figsize=(25,16))

p = sns.countplot(x='parental level of education', data = dataset, hue='Grade', palette='bright')

_ = plt.setp(p.get_xticklabels(), rotation=90) 