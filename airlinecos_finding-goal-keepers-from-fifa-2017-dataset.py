# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
players = pd.read_csv('../input/FullData.csv',parse_dates=True)
players['Height'] = players.Height.apply(lambda x: x.replace(' cm', '')).astype('int')
players['Weight'] = players.Weight.apply(lambda x: x.replace(' kg', '')).astype('int')
players = players.dropna(axis=1)
players[players.Preffered_Position == 'GK'].head()
selection_GK = [4,31,139,190]
players_GK_select = players.iloc[selection_GK]
players.Preffered_Position = np.where(players.Preffered_Position == 'GK', 0, 1)
players.Preffered_Position.head()
print('Goal Keeper player percentage: %.2f' %(100 * (players.shape[0]-players.Preffered_Position.sum())/players.shape[0]),'%')
selected_features = ['Height', 'Weight','Ball_Control', 'Dribbling', 'Marking', 'Sliding_Tackle',
       'Standing_Tackle', 'Aggression', 'Reactions', 'Attacking_Position',
       'Interceptions', 'Vision', 'Composure', 'Crossing', 'Short_Pass',
       'Long_Pass', 'Acceleration', 'Speed', 'Stamina', 'Strength', 'Balance',
       'Agility', 'Jumping', 'Heading', 'Shot_Power', 'Finishing',
       'Long_Shots', 'Curve', 'Freekick_Accuracy', 'Penalties', 'Volleys']
X = players[selected_features].values
X_GK_select = players_GK_select[selected_features].values
y = players.Preffered_Position.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)
X_GK_select_std = sc.transform(X_GK_select)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
X_GK_select_pca = pca.transform(X_GK_select_std)
plt.bar(range(pca.n_components_), pca.explained_variance_)
plt.xticks(range(pca.n_components_))
plt.ylabel('variance')
plt.xlabel('PCA feature')
plt.show()
plt.figure(figsize=([18,10]))
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], marker='.',alpha=0.3,s=100)
plt.scatter(X_GK_select_pca[:,0], X_GK_select_pca[:,1], 
            color=['black','orange','magenta','brown'], marker= 'o',s=100)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.show()
from matplotlib.colors import ListedColormap
import matplotlib.colors as colors

def plot_dr(X, y, clf, resolution=0.02):

    # setup marker generator and color map
    colors = ('blue', 'red')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = clf.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, 
                    c=cmap(idx), label=cl, marker='.',s=100)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1)
lr = lr.fit(X_train_pca, y_train)

plt.figure(figsize=([12,8]))
plot_dr(X_train_pca, y_train, clf=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()

plt.show()
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_pca, y_train)

plt.figure(figsize=([12,8]))
plot_dr(X_train_pca, y_train, clf=svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=0.0001, random_state=0)
svm.fit(X_train_pca, y_train)

plt.figure(figsize=([12,8]))
plot_dr(X_train_pca, y_train, clf=svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
from sklearn.svm import SVC

svm = SVC(kernel='linear', C=10.0, random_state=0)
svm.fit(X_train_pca, y_train)

plt.figure(figsize=([12,8]))
plot_dr(X_train_pca, y_train, clf=svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=1.0, random_state=0)
svm.fit(X_train_pca, y_train)

plt.figure(figsize=([12,8]))
plot_dr(X_train_pca, y_train, clf=svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_linear.png', dpi=300)
plt.show()
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=0.1, random_state=0)
svm.fit(X_train_pca, y_train)

plt.figure(figsize=([12,8]))
plot_dr(X_train_pca, y_train, clf=svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_linear.png', dpi=300)
plt.show()
from sklearn.svm import SVC

svm = SVC(kernel='rbf', C=10.0, random_state=0)
svm.fit(X_train_pca, y_train)

plt.figure(figsize=([12,8]))
plot_dr(X_train_pca, y_train, clf=svm)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper left')
plt.tight_layout()
# plt.savefig('./figures/support_vector_machine_linear.png', dpi=300)
plt.show()
