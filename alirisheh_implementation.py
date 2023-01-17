import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
dataFrame = pd.read_csv('../input/corona-virus-cases-homework-for-university/data.csv')
dataFrame
dataFrame['birth_year'].min()
dataFrame['birth_year'].max()
dataFrame['birth_year'].std()
print(dataFrame['region'].unique(), ' length : ', len(dataFrame['region'].unique()))
dataFrame.isnull()
dataFrame.dropna()
plt.subplot(131)

dataFrame['birth_year'].plot.hist()

plt.title('histogram from birth_year')

plt.subplot(132)

plt.title('birth_year box chart')

dataFrame['birth_year'].plot.box()

plt.subplot(133)

plt.title('correlation between birth_year and infected_by')

plt.scatter(x=dataFrame['infected_by'].to_numpy(), y=dataFrame['birth_year'].to_numpy())

fig = plt.gcf()

fig.set_size_inches(16, 8)

plt.rcParams.update({'font.size': 12})

plt.show()
plt.subplot(121)

plt.title('Bar chart for regions')

dataFrame['region'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Line chart for regions')

dataFrame['region'].value_counts().plot(kind='line')

fig = plt.gcf()

fig.set_size_inches(16, 8)

plt.rcParams.update({'font.size': 12})

plt.show()
plt.subplot(121)

plt.title('Bar chart for infection reason')

dataFrame['infection_reason'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Line chart for infection reason')

dataFrame['infection_reason'].value_counts().plot(kind='line')

fig = plt.gcf()

fig.set_size_inches(16, 8)

plt.rcParams.update({'font.size': 12})

plt.show()
plt.subplot(121)

plt.title('Bar chart for sex distribution')

dataFrame['sex'].value_counts().plot(kind='bar')

plt.subplot(122)

plt.title('Correlation between sex and age')

plt.scatter(x=dataFrame['sex'].to_numpy(), y=dataFrame['birth_year'].to_numpy())

fig = plt.gcf()

fig.set_size_inches(16, 8)

plt.rcParams.update({'font.size': 12})

plt.show()
outlier_count = dataFrame.count() - dataFrame[np.abs(dataFrame.birth_year-dataFrame.birth_year.mean()) <= (3*dataFrame.birth_year.std())].count()

print(outlier_count)
q_birth_year = dataFrame["birth_year"].quantile(0.99)

q_infected_by = dataFrame['infected_by'].quantile(1) # I changed this from 0 to 1

outlier_count_birth_year = dataFrame.count() - dataFrame[dataFrame.birth_year < q_birth_year].count()

outlier_count_infected_by = dataFrame.count() - dataFrame[dataFrame.infected_by < q_infected_by].count()

print("By birth \n", outlier_count_birth_year)

print("By infected \n", outlier_count_infected_by)