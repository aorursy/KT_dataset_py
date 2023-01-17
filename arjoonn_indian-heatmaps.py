import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%pylab inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/all.csv', index_col=0)

df.info()
plt.figure(figsize=(10, 7))

ax = plt.gca()

sns.boxplot(x='State', y='Persons', data=df, linewidth=1)

ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)

plt.title('Population in the "States". <pun intended>')

#plt.savefig('population_per_state.png')
education_cols = ['Below.Primary', 'Primary', 'Middle', 'Matric.Higher.Secondary.Diploma',

                'Graduate.and.Above']

temp = df[education_cols + ['State']].groupby('State').sum()



plt.figure(figsize=(4, 7))

sns.heatmap(np.round(temp.T / temp.sum(axis=1), 2).T, cmap='gray_r',

            linewidths=0.01, linecolor='white', annot=True)

plt.title('Which state has what fraction of people in what bracket?')
age_cols = ['X0...4.years','X5...14.years',

            'X15...59.years','X60.years.and.above..Incl..A.N.S..']

temp = df[age_cols+['State']].groupby('State').sum()



plt.figure(figsize=(15, 3))

ax = plt.gca()

temp.columns=['0 to 4 years', '5 to 14 years', '15 to 59 years', '60 years +']

sns.heatmap(np.round(temp / temp.sum(axis=0), 2).T, linecolor='white',

            linewidths=0.01, cmap='gray_r', annot=True)

ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)

plt.title('Age group distribution over states')
worker_cols = ['Main.workers', 'Marginal.workers', 'Non.workers']

temp = df[worker_cols+['State']].groupby('State').sum()



plt.figure(figsize=(5, 7))

temp2 = temp.T / temp.sum(axis=1) # What fraction of the group is in what state

sns.heatmap(np.round(temp2 / temp2.sum(axis=0), 2).T, linecolor='white',

            linewidths=0.01, cmap='gray_r', annot=True)

plt.title('Working class distribution over states. Rows sum to 1')
religion_cols = ['Religeon.1.Name','Religeon.1.Population',

                 'Religeon.2.Name','Religeon.2.Population',

                 'Religeon.3.Name','Religeon.3.Population']

temp = df[religion_cols + ['State']].copy()

for i in '123':

    temp['Religeon.'+i+'.Name'] = temp['Religeon.'+i+'.Name'].str.split('.').str[-1]

temp2 = pd.DataFrame([], columns=['Name', 'Population', 'State'])

for i in '123':

    a = temp[['Religeon.'+i+'.Name', 'Religeon.'+i+'.Population', 'State']].copy()

    a.columns = ['Name', 'Population', 'State']

    temp2 = pd.concat([a, temp2])

grouped = temp2.groupby(['State', 'Name']).sum()

temp2 = grouped.reset_index().fillna(1)

ct = pd.crosstab(temp2.State, temp2.Name, temp2.Population, aggfunc=np.sum)

ct = ct.fillna(1)

plt.figure(figsize=(7, 7))

sns.heatmap(np.round(ct / ct.sum(axis=0), 2), cmap='gray_r', linecolor='black',

            linewidths=0.01, annot=True)

plt.title('Which religion resides in which state? (Columns sum to 1)')
plt.figure(figsize=(7,7))

sns.heatmap(np.round(ct.T / ct.sum(axis=1), 2).T, cmap='gray_r',

            linecolor='black', linewidths=0.01, annot=True)

plt.title('States have what fraction of which religion?(Rows sum to 1)')