import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sklearn.preprocessing as preprocessing



df = pd.read_excel('../input/State Of JavaScript 2016 (clean).xlsx')
# clean Favorite Editor

df['Favorite Text Editor'].fillna(df['Other'], inplace=True)

editor = df['Favorite Text Editor'].str.lower()

editor.replace(regex='.*code.*', value='visual studio [code]', inplace=True)

editor.replace(regex='.*visual.*', value='visual studio [code]', inplace=True)

editor.replace(regex='.*vs.*', value='visual studio [code]', inplace=True)

editor.replace(regex='.*idea.*', value='intellij idea', inplace=True)

editor.replace(regex='.*intellij.*', value='intellij idea', inplace=True)



df['Favorite Text Editor'] = editor

editor.value_counts()[0:10].plot.bar()

plt.show()
df['Yearly Salary'].value_counts().plot.bar()

plt.show()
df['Years of Experience'].value_counts().plot.bar()

plt.show()
df.loc[df['Years of Experience']=='20+ years', 'Favorite Text Editor'].value_counts()[0:10].plot.bar()

plt.show()
df.loc[df['Yearly Salary']=='$200k+', 'Favorite Text Editor'].value_counts()[0:10].plot.bar()

plt.show()