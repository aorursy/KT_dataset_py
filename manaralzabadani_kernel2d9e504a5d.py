import numpy as np



A1=np.arange(1,13).reshape(3,4)

A2=A1**3

print(A2)
import pandas as pd  



exam_data = {'name': ['Anastasia', 'Dima', 'Katherine',

'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin',

'Jonas'],

'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8,

19],

'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes',

'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



data = pd.DataFrame(exam_data,labels)

print(data.iloc[0:3])

print(".....................................................")

print(data[['name', 'score']])
import numpy as np



A1=np.arange(2,11).reshape(3,3)

print(A1)