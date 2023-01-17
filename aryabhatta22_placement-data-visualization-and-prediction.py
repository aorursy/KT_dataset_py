import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
df.info()
df.isnull().any()
df['status'].unique()
df[df['status'] == 'Not Placed']['salary'].unique() # column has only null values
df['salary'].fillna(0.0, inplace = True)
df.isnull().any()
df.head()
df['Mean Score'] = df['ssc_p'] + df ['hsc_p'] + df['degree_p'] +df['mba_p']

df['Mean Score'] = df['Mean Score'] / 4
sns.distplot(df['salary'], kde = False)

plt.xlabel('salary')

plt.ylabel('number of students')

plt.title('Number of students and salary range')
placed = df[df['status'] == 'Placed']

unplaced =df[df['status'] == 'Not Placed']
sns.catplot(x="status", kind="count", data=df);
sns.catplot(x="status", kind="count",hue ='gender', data=df);
ax = plt.subplot(111)

sns.scatterplot(x='Mean Score',y='salary',hue='gender',data= placed)

ax.legend(bbox_to_anchor=(1.3, 1.0))
sns.catplot(x="workex", kind="count",hue ='gender', data=df, col='status');
sns.catplot(x="specialisation", kind="count", data=df, col='status');
spec = np.asarray(df['specialisation'].unique())

placedSpec = list(map(lambda spec: len(placed[placed['specialisation'] == spec]), spec))

plt.pie(x = placedSpec, shadow = True , labels = spec, radius = 1.5, startangle=90)

plt.title('Placed')

plt.show()

unplacedSpec = list(map(lambda spec: len(unplaced[unplaced['specialisation'] == spec]), spec))

plt.pie(x = unplacedSpec, shadow = True , labels = spec, radius = 1.5, startangle=90)

plt.title('Not placed')

plt.show()
ax = plt.subplot(111)

sns.scatterplot(x='Mean Score',y='salary',hue='specialisation',data= placed)

ax.legend(bbox_to_anchor=(1.4, 1.0))
placed.head()
sns.catplot(x="degree_t", kind="count", data=df, col ='status');
degree = np.asarray(df['degree_t'].unique())

placedDegree = list(map(lambda deg: len(placed[placed['degree_t'] == deg]), degree))

plt.pie(x = placedDegree, shadow = True , labels = degree, radius = 1.5 )

plt.title('Placed')

plt.show()



unplacedDegree = list(map(lambda deg: len(unplaced[unplaced['degree_t'] == deg]), degree))

plt.pie(x = unplacedDegree, shadow = True , labels = degree, radius = 1.5 )

plt.title('Not placed')

plt.show()
ax = plt.subplot(111)

sns.scatterplot(x='Mean Score',y='salary',hue='degree_t',data= placed)

ax.legend(bbox_to_anchor=(1.4, 1.0))
sns.jointplot(x='etest_p', y = 'salary', data = placed , kind="hex", color="#4CB391")
sns.jointplot(x='Mean Score', y = 'etest_p', data = placed , kind="hex", color="#4CB391")
sns.jointplot(x='degree_p', y = 'salary', data = placed , kind="hex", color="#4CB391")
sns.jointplot(x='Mean Score', y = 'salary', data = placed , kind="hex", color="#4CB391")
df.head()
sns.catplot(x="workex", kind="count", data=df, col ='status');
sns.catplot(x="workex", kind="count", data=df, hue = 'gender');
df.head()
scoresMatPlaced = placed.loc[:, ['ssc_p','hsc_p', 'degree_p', 'etest_p', 'mba_p', 'salary', 'Mean Score']]
scoresMatPlaced.head()
sns.heatmap(scoresMatPlaced.corr(), cmap = 'coolwarm')
sns.clustermap(scoresMatPlaced.corr(), cmap ='coolwarm')