import pandas as pd

import numpy as np
a = [['Yes','Nomobile','cat'],['No','Mobile','dog'],['Yes','Mobile','Bird'],['Yes','Mobile','dog']]

df = pd.DataFrame(a)

print(df)
df[0] = df[0].replace(to_replace=['No', 'Yes'], value=[0, 1])

df[1] = df[1].replace(to_replace=['Nomobile', 'Mobile'], value=[0, 1])

print(df)
category = pd.unique(df[2])

print(category)
for i in range(0,len(category)):

    judge_list = list()

    for j in range(0,len(df)):

        if df[2][j] == category[i]:

            judge_list.append(1)

        else:

            judge_list.append(0)

    df[category[i]]= judge_list
print(df)