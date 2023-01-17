import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
stud_data = pd.read_csv("../input/stud_data.csv")

stud_data
stud_data.sort_values('Stud_Roll')
stud_data[stud_data.isna().any(axis=1)]
stud_data.fillna(0, inplace=True)
q1 = np.percentile(stud_data['Marks_English'], 25)

q3 = np.percentile(stud_data['Marks_English'], 75)



IQR = q3 - q1



least = q1 - 1.5*q3

maxi = q3 + 1.5*q3
stud_data.loc[stud_data['Marks_English'] < least]
stud_data.loc[stud_data['Marks_English'] > maxi]
plt.hist(stud_data['Marks_English'])

plt.show()
plt.scatter(stud_data["Marks_English"], stud_data['Marks_Maths'])

plt.show()