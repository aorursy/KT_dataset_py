import pandas as pd
import numpy as np

student_data  = {'name': ['Ankita', 'Seema', 'Pratik', 'Suman', 'Disha', 'Anchal', 'Swamay'],
        'marks': [98, np.nan, 80, np.nan, 91, 85, 87],
        'attempts': [1, 0, 2, 0, 2, 1, 5],
        'qualification status': ['yes', 'no', 'no', 'no', 'yes', 'no', 'yes']}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

df = pd.DataFrame(student_data , index=labels)
print("First three rows of the data frame:")
print(df.iloc[:3])
from matplotlib import pyplot as plt

TEAMS=['CSK','KKR','DC','MI']

 
SCORE = [110,99,101,150]
  
tick_label = ['CSK','KKR','DC','MI'] 
  
plt.bar(TEAMS, SCORE, tick_label = tick_label, width = 0.6, color = ['green', 'green','green','RED',]) 
  
plt.xlabel('TEAMS') 
plt.ylabel('SCORE')  
plt.title('SCORE GRAPH')

import numpy as np
a=np.array([1,2,3,4,5])
print("Array 1: ",a)
b=np.array([1,3,8])
print('Array 2: ', b)
print("Common values between the two arrays: ")
print(np.intersect1d(a,b))

for i,val in enumerate(a):
    if val in b:
        a=np.delete(a, np.where(a==val)[0][0])
        for i, val in enumerate(b):
            if val in a:
                a=np.delete(a, np.where(a==val)[0][0])
                print("Arrays after deletion of common elements: ")
                print(a)
                print(b)