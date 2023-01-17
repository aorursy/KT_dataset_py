import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import math





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



train=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")
print(train.columns)

print(train.values[0])

print(train.shape) 

print(train.count(axis='index'))



print("\n")



print(test.columns)

print(test.values[0])

print(test.shape)

print(test.count(axis='index'))
total = train.shape[0]

sex_count = train.groupby(['Sex']).count()['PassengerId']

survived = train.groupby(['Survived']).count()['PassengerId']

survived_by_sex = train.groupby(['Sex','Survived']).count()['PassengerId']

print("Total number of people: "+str(total))

print(sex_count)

print(survived)

print(survived_by_sex)
males_survived = survived_by_sex[3]

females_survived = survived_by_sex[1]

males = sex_count[1]

females = sex_count[0]



print("Probablity of males surviving: "+str(males_survived/males))

print("Probablity of females surviving: "+str(females_survived/females))
ids = test['PassengerId']

sex = test['Sex']

prediction = []

for i in sex:

    if(i == 'male'):

        prediction.append(0)

    else:

        prediction.append(1)

simple_sol = pd.DataFrame({ 'PassengerId':ids,'Survived':prediction })

simple_sol.to_csv('simple_sol.csv', index=False)

#output.to_csv('submission.csv', index=False)
age_grp = []

for age in train['Age']:

    if(math.isnan(age)):

        age_grp.append('Not Defined')

    elif(age<14):

        age_grp.append('Child')

    else:

        age_grp.append('Adult')

    

train['Age Group'] = age_grp

age_survived = train.groupby(['Age Group','Survived']).count()['PassengerId']

print(age_survived)
survival_graph = train.groupby(['Age Group','Sex','Pclass','Survived']).count()['PassengerId']

print(survival_graph)