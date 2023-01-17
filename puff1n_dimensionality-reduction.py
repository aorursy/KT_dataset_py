import numpy as np 

import pandas as pd 



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn-dark')



from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB



from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.decomposition import KernelPCA
df = pd.read_csv('/kaggle/input/breast-cancer-data/data.csv')

df = df.drop('Unnamed: 32', axis=1)

df['id'] = df['id'].astype('category')

df.info()
corr=df.corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=False, linewidths=.5, cbar_kws={"shrink": .5});
X = df.drop(['diagnosis', 'id'], axis=1)

y = df['diagnosis']

X.shape
X_transformed = PCA(n_components=3).fit_transform(X)

X_transformed_ker = KernelPCA(n_components=3).fit_transform(X)

X_tsne = TSNE(n_components=3, perplexity=15).fit_transform(X)
def get_slices_scatter(X_transformed, y=y):

    fig, ax = plt.subplots(1, 3, figsize=(20, 4))

    sns.scatterplot(x=X_transformed[:, 0], y=X_transformed[:, 1], ax=ax[0], hue=y)

    sns.scatterplot(x=X_transformed[:, 0], y=X_transformed[:, 2], ax=ax[1], hue=y)

    sns.scatterplot(x=X_transformed[:, 1], y=X_transformed[:, 2], ax=ax[2], hue=y)

    

def get_3d_scatter(X_transformed, lab, y=y):

    trace = go.Scatter3d(x = X_transformed[:, 0], y = X_transformed[:, 1],

                     z = X_transformed[:, 2],mode = 'markers',

                     marker = dict(size = 2, color = y.map({'M': 'coral', 'B': 'blue'}))

                      )



    data = [trace]

    layout = go.Layout(title=lab)

    fig = go.Figure(data=data, )

    iplot(fig)

    



# get_slices(X_transformed)

get_3d_scatter(X_transformed, lab='PCA')

get_3d_scatter(X_transformed_ker, lab='KernelPCA')

get_3d_scatter(X_tsne, lab='TSNE')

lr = LogisticRegression(penalty='l1', solver='saga', max_iter=100000)

dt = DecisionTreeClassifier()

nb = GaussianNB()

svm = SVC()



lr_test_acc = []

dt_test_acc = []

nb_test_acc = []

svm_test_acc = []





kf = StratifiedKFold(n_splits=9, random_state=666, shuffle=True)

for train_i, test_i in kf.split(X, y):

    X_train, X_test = X_transformed[train_i], X_transformed[test_i]

    y_train, y_test = y[train_i], y[test_i]

    

    lr.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)

    lr_test_acc.append(accuracy_score(y_test,lr_pred))

    

    dt.fit(X_train, y_train)

    dt_pred = dt.predict(X_test)

    dt_test_acc.append(accuracy_score(y_test,dt_pred))

    

    nb.fit(X_train, y_train)

    nb_pred = nb.predict(X_test)

    nb_test_acc.append(accuracy_score(y_test,nb_pred))

    

    svm.fit(X_train, y_train)

    svm_pred = svm.predict(X_test)

    svm_test_acc.append(accuracy_score(y_test,svm_pred))  

    

plt.plot(lr_test_acc, label='LogR')

plt.plot(dt_test_acc, label='DecT')

plt.plot(nb_test_acc, label='NB')

plt.plot(svm_test_acc, label='SVM')

plt.ylim(0.75, 1.)

plt.xlabel('N Fold')

plt.ylabel('Accuracy')

plt.legend()