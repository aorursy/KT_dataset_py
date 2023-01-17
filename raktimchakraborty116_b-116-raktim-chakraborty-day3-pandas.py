#question 1

import pandas as pd

import numpy as np



df = pd.DataFrame({'X':[78,85,96,80,86], 'Y':[84,94,89,83,86],'Z':[86,97,96,72,83]});

print(df)
#question 2



exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data , index=labels)

print(df)

#question 3



df = pd.DataFrame(exam_data , index=labels)

print("First three rows of the data frame:")

print(df.iloc[:3])
#question 4



df = pd.read_csv("../input/titanic/train_and_test2.csv")

print("Select specific columns and rows:")

print(df.iloc[[1, 3, 5, 6, 8], [1, 3, 5]])
#question 5



df = pd.DataFrame(exam_data , index=labels)

print("Rows where score is missing:")

print(df[df['score'].isnull()])