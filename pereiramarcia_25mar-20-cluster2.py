# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# carregando



df = pd.read_csv('/kaggle/input/insurance/insurance.csv')

df.shape
df.head()
# verificando o dataframe



df.info()
# analisando os dados



df['charges'].plot.hist(bins=50)
df['bmi'].plot.hist(bins=50)
# verificando a idade



df['age'].min(), df['age'].max()
# criar categorias para idades



def age_to_cat(age):

    if (age >= 18) & (age <= 35):

        return "Adulto"

    elif (age > 35) & (age <= 55):

        return "Senior"

    else:

        return "Idoso"



df['age_cat'] = df['age'].apply(age_to_cat)



df.head()
# visualizando



import seaborn as sns



# relação idade x gastos



sns.stripplot(data=df, x='age_cat', y= "charges", linewidth=1)
sns.stripplot(data=df, x='smoker', y= "charges", linewidth=1)
sns.stripplot(data=df, x='smoker', y= "charges", linewidth=1, hue='age_cat')
# dever de casa



# usando bmi e charges
from sklearn.cluster import KMeans
df1 = df[['bmi', 'charges']]
df1.head().T
# o próprio k-mean já tem a propriedade de inertia que calcula o sse

sse= []

for k in range (1, 15):

    kmeans = KMeans(n_clusters=k, random_state=42).fit(df1)

    sse.append(kmeans.inertia_)
sse
import matplotlib.pyplot as plt



plt.plot(range(1, 15), sse, 'bx-')

plt.title('Elbow Method')

plt.xlabel('Numero de cluster')

plt.ylabel('SSE')

plt.show()
# vamos fazer com 4 clusters



kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)

cluster_id = kmeans.fit_predict(df1)
cluster_id
# vamos guardar os resultados como uma coluna no dataframe



df1['cluster_id'] = cluster_id



df1.sample(10).T
# Identificando as ocorrências do cluster 0



df1[df1['cluster_id']== 0].describe()
import seaborn as sns
sns.scatterplot(data=df1, x='bmi', y='charges', hue='cluster_id')