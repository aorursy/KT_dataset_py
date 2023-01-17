# Відображення графіків у Jupyter notebook

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

# формат графіків - svg 

%config InlineBackend.figure_format = 'svg' 



# Збільшуємо стандартний розмір графіків

from pylab import rcParams

rcParams['figure.figsize'] = 8,5

# Імпортуємо Pandas i Numpy

import pandas as pd

import numpy as np
df = pd.read_csv("../input/telecom-churn/telecom_churn.csv")
df.head()
df.shape
df.info()
df['churn'].value_counts()
df['churn'].value_counts().plot(kind='bar', label='churn')

plt.legend()

plt.title('Розподіл відтоку клієнтів');
corr_matrix = df.drop(['state', 'international plan', 'voice mail plan',

                      'area code'], axis=1).corr()
sns.heatmap(corr_matrix);
features = list(set(df.columns) - set(['state', 'international plan', 'voice mail plan',  'area code',

                                       'churn', 'phone_number', 'total day charge', 'total eve charge', 

                                       'total night charge','total intl charge','phone number']))
df[features].hist(figsize=(20,12));
# Для запуску розкоментуйте цей код

#sns_pairplot = sns.pairplot(df[features + ['churn']], hue='churn');

#ns_pairplot.savefig('pairplot.png')
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 10))



for idx, feat in  enumerate(features):

    sns.boxplot(x='churn', y=feat, data=df, ax=axes[int(idx / 4), idx % 4])

    axes[int(idx / 4), idx % 4].legend()

    axes[int(idx / 4), idx % 4].set_xlabel('churn')

    axes[int(idx / 4), idx % 4].set_ylabel(feat)

fig.savefig('boxplot.png')
_, axes = plt.subplots(1, 2, sharey=True, figsize=(16,6))



sns.boxplot(x='churn', y='total day minutes', data=df, ax=axes[0]);

sns.violinplot(x='churn', y='total day minutes', data=df, ax=axes[1]);
sns.countplot(x='customer service calls', hue='churn', data=df);
_, axes = plt.subplots(1, 2, sharey=True, figsize=(16,6))



sns.countplot(x='international plan', hue='churn', data=df, ax=axes[0]);

sns.countplot(x='voice mail plan', hue='churn', data=df, ax=axes[1]);
df.groupby(['state'])['churn'].agg([np.mean]).sort_values(by='mean', ascending=False).T
from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler
df.head()
# Перетворимо ознаки international plan,voice mail plan в числові. 

# Відкинемо колонки: churn, state, phone number

X = df.drop(['churn', 'state','phone number'], axis=1)

X['international plan'] = pd.factorize(X['international plan'])[0]

X['voice mail plan'] = pd.factorize(X['voice mail plan'])[0]
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
%%time

tsne = TSNE(random_state=17)

tsne_representation = tsne.fit_transform(X_scaled)
rcParams['figure.figsize'] = 10,7

#plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1]);
#plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1], 

       # c=df['churn'].map({False: 'blue', True: 'orange'}));
_, axes = plt.subplots(1, 2, sharey=True, figsize=(16,6))



axes[0].scatter(tsne_representation[:, 0], tsne_representation[:, 1], 

            c=df['international plan'].map({'yes': 'blue', 'no': 'orange'}));

axes[1].scatter(tsne_representation[:, 0], tsne_representation[:, 1], 

            c=df['voice mail plan'].map({'yes': 'blue', 'no': 'orange'}));

axes[0].set_title('international plan');

axes[1].set_title('voice mail plan');