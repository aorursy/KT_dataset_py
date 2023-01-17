import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df=pd.DataFrame({

    'A':[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],

    'B':[10,13,11,16,37,67,21,19,23,20,45,24,67,10,15,102,202,56,178,112]

})

df
plt.scatter(df['A'],df['B'])
df.describe()
q1=df.quantile(.25, axis = 0) 

q3=df.quantile(.75, axis = 0) 

print("1st (25%) Quantile:\n"+str(q1))

print("\n3rd (75%) Quantile:\n"+str(q1))

icq=q3-q1

print("\nInter Quantile Range:\n"+str(icq))
icq_applied=icq[1]

print("Outliers to be detected for Parameter B")

print("Inter Quantile Range for B is",icq_applied)

lb=q1[1]-(1.5*icq_applied)

print(lb)

ub=q3[1]+(1.5*icq_applied)

print(ub)
df1=df[(df['B']>lb)&(df['B']<ub)]

print("Final Dataframe after removal of Outliers:")

df1
print("Length of Actual Dataframe: ",len(df))

print("Length of Dataframe after removal of outliers: ",len(df1))
plt.scatter(df1['A'],df1['B'])