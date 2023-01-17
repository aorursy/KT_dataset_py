import pandas as pd

import seaborn as sns

import sklearn.metrics

import matplotlib.pyplot as plt

import numpy as np

import plotly.plotly as py

import plotly.graph_objs as go

import warnings



from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler



warnings.filterwarnings("ignore")
data = pd.read_csv('../input/StudentsPerformance.csv')
print("Colunas: {0}\n".format(data.columns))
data = data.rename(columns={'race/ethnicity' : 'ethnicity',

                            'parental level of education' : 'parentalLevelEducation',

                            'test preparation course' : 'preparationCourse',

                            'math score' : 'mathScore',

                            'reading score' : 'readingScore',

                            'writing score': 'writingScore'})



print("Colunas: {0}\n".format(data.columns))
print(data.info())
print("Informações sobre a tabela\n{0}".format(data.describe(include='all')))
print("Formato do conjunto de dados {0}".format(data.shape))
data.head()
sns.pairplot(data, kind="scatter", hue="gender")

plt.show()
sns.pairplot(data, kind="scatter", hue="ethnicity")

plt.show()
sns.pairplot(data, kind="scatter", hue="parentalLevelEducation")

plt.show()
sns.pairplot(data, kind="scatter", hue="preparationCourse")

plt.show()
fig, axs = plt.subplots(1, 5, figsize=(16, 5), sharey=True)

sns.countplot(data.gender, ax = axs[0])

ax0 = sns.countplot(data["ethnicity"], ax = axs[1])

ax0.set_xticklabels(ax0.get_xticklabels(), rotation=90)

ax1 = sns.countplot(data["parentalLevelEducation"], ax = axs[2])

ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

sns.countplot(data.lunch, ax = axs[3])

sns.countplot(data["preparationCourse"], ax = axs[4])
scores = data.loc[:,["mathScore","readingScore","writingScore"]]
wcss = []

print('Valores do WCSS:')

for i in range(1, 11):

    kmeans = KMeans(n_clusters = i, random_state = 0)

    kmeans.fit(scores)

    print(kmeans.inertia_)

    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)

plt.xlabel('Número de clusters')

plt.ylabel('WCSS')
kmeans = KMeans(n_clusters = 5, random_state = 0)

previsoes = kmeans.fit_predict(scores)

data['class'] = previsoes

print(data)

plt.show()
sns.pairplot(data, hue='class', diag_kind=None)
data.loc[data['class'] == 2, 'grade'] = "A"

data.loc[data['class'] == 0, 'grade'] = "B"

data.loc[data['class'] == 1, 'grade'] = "C"

data.loc[data['class'] == 3, 'grade'] = "D"

data.loc[data['class'] == 4, 'grade'] = "F"



sns.pairplot(data, hue='grade', diag_kind='hist')
fig, axs = plt.subplots()

sns.countplot(data.grade)