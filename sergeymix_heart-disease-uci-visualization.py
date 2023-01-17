import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



plt.rcParams["figure.figsize"] = (12, 8)
data = pd.read_csv('../input/heart-disease-uci/heart.csv')

data.head(5)
# Посмотрим на наличие пропусков в данных



data.info()
# Посмотрим на статистики по столбцам



data.describe()
# Количество пациентов с таргетным значением 1 или 0



data['target'].value_counts()
n_bins = 10

healthy = data[data['target'] == 0]['age'].values

ill = data[data['target'] == 1]['age'].values





plt.hist([healthy, ill], n_bins, histtype='bar', color=['C5', 'C9'], alpha=0.7, normed='density');

plt.legend(('healthy', 'ill'), fontsize='large');

plt.title('Distribution of patients on age in two groups', fontdict={'fontsize': 18});
features = data.columns[1:-1]

n_bins = 10

fig, axes = plt.subplots(4, 3, figsize=(20, 15), sharex=False)



for i, ax in enumerate(axes.ravel()):

    healthy = data[data['target'] == 0][features[i]].values

    ill = data[data['target'] == 1][features[i]].values

    ax.hist([healthy, ill], n_bins, histtype='bar', color=['C5', 'C9'], alpha=0.7, normed='density');

    

    #if i % 3 == 0:

    #    ax.set_ylabel("density", size=10)

    

    # ax.set_xlabel("feature")

    ax.set_title(features[i], size=14)

    ax.legend(('healthy', 'ill'), fontsize='medium');

    

# fig.suptitle("Densities for different features", size=20, y=1.025)

fig.tight_layout()
features_quantitative = ['trestbps', 'chol', 'thalach', 'oldpeak']

sns.set(style="whitegrid")



fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))



for i, ax in enumerate(axes.ravel()[:-2]):

    

    sns.boxplot(y=data[features_quantitative[i]], x=data["target"], palette="Set3", ax=ax);

    ax.set_title(features_quantitative[i], size=14)

    # plt.show()



fig.tight_layout()
fig, ax = plt.subplots(figsize=(14,9))

sns.heatmap(data.drop(['target'], axis=1).corr(method='pearson'), annot=True, fmt='.2f', cmap="YlGnBu", ax=ax);
sns.pairplot(data.drop(['target'], axis=1), palette="YlGnBu");
sns.pairplot(data[features_quantitative + ['target']], hue='target', palette="YlGnBu", height=4, plot_kws={"s": 50});
from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler



X = data.drop(['target'], axis=1)



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
%%time

tsne = TSNE(random_state=17)

tsne_representation = tsne.fit_transform(X_scaled)
fig, ax = plt.subplots(figsize=(12,8))



scatter = ax.scatter(tsne_representation[:, 0], tsne_representation[:, 1], c=data['target'].map({0: 'palegreen', 1: 'royalblue'}));



ax.set_title('Two-dimensional projection of features', fontdict={'fontsize': 20}, pad=20)

ax.set_xlabel("new_feature_1")

ax.set_ylabel("new_feature_2");