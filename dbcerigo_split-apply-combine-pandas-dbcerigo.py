import numpy as np # linear algebra

import pandas as pd # data processing, 



df = pd.read_csv('../input/train.csv')

df = df.drop(columns=['Ticket', 'Cabin', 'PassengerId', 'SibSp', 'Parch'])
print(len(df))

df.head()
class Q1:

    @staticmethod

    def hint():

        print('HINT: You can compute the survival rate by taking the average of the `Survived` column')

    @staticmethod

    def answer():

        print('ANSWER: male: 0.188908, female: 0.742038')

    @staticmethod

    def solution():

        print('SOLUTION: df.groupby(\'Sex\').Survived.mean()')
# write your solution here...
#Q1.hint()

#Q1.answer()

#Q1.solution()
class Q2:

    @staticmethod

    def hint():

        print('HINT: You can sort a series by using the `sort_values()` function!')

    @staticmethod

    def answer():

        print('''

ANSWER: 

Pclass

1    0.629630

2    0.472826

3    0.242363\n''')

    @staticmethod

    def solution():

        print('SOLUTION: df.groupby(\'Pclass\').Survived.mean().sort_values(ascending=False)')
# write your solution here...
#Q2.hint()

#Q2.answer()

#Q2.solution()
class Q3:

    @staticmethod

    def hint():

        print('HINT: Don\'t forget about the `max` and `min`  aggregators!')

    @staticmethod

    def answer():

        print('''

ANSWER: 

Pclass

1    512.3292

2     73.5000

3     69.5500\n''')

    @staticmethod

    def solution():

        print('SOLUTION: df.groupby(\'Pclass\').Fare.max()')
# write your solution here...
#Q3.hint()

#Q3.answer()

#Q3.solution()
class Q4:

    @staticmethod

    def hint():

        print('HINT: `groupby` can also take a list!')

    @staticmethod

    def answer():

        print('''

ANSWER: 

Embarked  Pclass

C         1         0.694118

Q         2         0.666667

S         1         0.582677

C         2         0.529412

Q         1         0.500000

S         2         0.463415

C         3         0.378788

Q         3         0.375000

S         3         0.189802\n''')

    @staticmethod

    def solution():

        print('SOLUTION: df.groupby([\'Embarked\', \'Pclass\']).Survived.mean().sort_values(ascending=False)')
# write your solution here...
#Q4.hint()

#Q4.answer()

#Q4.solution()