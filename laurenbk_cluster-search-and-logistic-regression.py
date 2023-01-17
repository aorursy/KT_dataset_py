# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/FIFA 2018 Statistics.csv')
df.head()
df.info()
df['Own goals'] = df['Own goals'].fillna(0)
df['1st Goal'] = df['1st Goal'].fillna(-1)
features = df._get_numeric_data().astype(np.float64)
features = features.drop(columns=['Own goal Time'])
print(features.columns)
plt.figure(figsize=(20,20))
sns.pairplot(df, vars=features.columns, hue='Man of the Match')
sns.heatmap(features.corr())
rounds = pd.get_dummies(df['Round'])
df = pd.concat([df, rounds], axis=1)

scaler = preprocessing.MinMaxScaler()
features_norm = scaler.fit_transform(features)
pca = PCA(n_components=2)
features_pca = pca.fit_transform(features_norm)
man_of_the_match_map = {
    'No': 'cornflowerblue',
    'Yes': 'darkorange'
}
for k,v in man_of_the_match_map.items():
    pca_sub = features_pca[df['Man of the Match']==k]
    plt.scatter(pca_sub[:,0], pca_sub[:,1], 
                c=v, label=k)
    plt.legend()
features_tsne = TSNE(n_components=2, 
                     perplexity=30, 
                     learning_rate=5).fit_transform(
                         features_norm)
for k,v in man_of_the_match_map.items():
    tsne_sub = features_tsne[df['Man of the Match']==k]
    plt.scatter(tsne_sub[:,0], tsne_sub[:,1], 
                c=v, label=k)
    plt.legend()
man_of_the_match_map = {
    'No': 'Blues',
    'Yes': 'Reds'
}
plt.figure(figsize=(10,5))
i = 1
for k,v in man_of_the_match_map.items():
    ax = plt.subplot(1, 2, i )
    tsne_sub = features_tsne[df['Man of the Match']==k]
    ax.hexbin(tsne_sub[:,0], tsne_sub[:,1], gridsize=15, label=k, cmap=v)
    i += 1
def logreg(f_norm, motm):
    x_train, x_test, y_train, y_test = train_test_split(f_norm, motm,
                                                        train_size=0.7, 
                                                        test_size=0.3, 
                                                        random_state=1)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    logistic = LogisticRegression(n_jobs=1)
    features_lreg = logistic.fit(x_train, y_train)
    predictions = features_lreg.predict(x_test)
    print(f'{sum(predictions == y_test)}/{len(y_test)} predictions were correct')
    return x_train, x_test, y_train, y_test, predictions
_ = logreg(features_norm, df['Man of the Match'])
print(classification_report(y_test, predictions))
features_subset = features.drop(columns=['Passes', 'Attempts', 'Own goals'])
scaler = preprocessing.MinMaxScaler()
fsub_norm = scaler.fit_transform(features_subset)
x_train, x_test, y_train, y_test, predictions = logreg(features_subset, df['Man of the Match'])
print(classification_report(y_test, predictions))

