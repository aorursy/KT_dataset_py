import pandas as pd

from matplotlib import pyplot as plt
df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

df.head()
df.replace(to_replace='male', value=0, inplace=True)

df.replace(to_replace='female', value=1, inplace=True)

df.replace(to_replace=['group A', "group B", "group C", "group D", "group E"], value=[1,2,3,4,5], inplace=True)

df.replace(to_replace=["bachelor's degree", 'some college', "master's degree", 

                       "associate's degree", 'high school', 'some high school'],

                        value=[5,3,6,4,2,1], inplace=True)

df.replace(to_replace=['standard', 'free/reduced'], value=[1,2], inplace=True)

df.replace(to_replace=['none', 'completed'], value=[0,1], inplace=True)
df.head()
plt.scatter(df['math score'],df['parental level of education'])

plt.scatter(df['writing score'],df['parental level of education'])

plt.scatter(df['reading score'],df['parental level of education'])

plt.legend(['Math Score', 'Writing Score', 'Reading Score'])

plt.vlines(x=40, ymin=0, ymax=7, linestyles='dashed')

plt.xlabel('Marks')

plt.ylabel('Parental Education')

plt.show()
fig = plt.figure(dpi=150)

ax = fig.add_subplot(111)

cax = ax.matshow(df.corr(), vmin=-0.5, vmax=1)

fig.colorbar(cax)

ticks = range(0,8)

ax.set_xticks(ticks)

plt.xticks(rotation=90)

ax.set_yticks(ticks)

ax.set_xticklabels(df.columns)

ax.set_yticklabels(df.columns)

plt.show()