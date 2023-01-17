import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
from importlib import reload
%matplotlib inline
df = pd.read_csv("../input/ph-data.csv")
df = shuffle(df)
df.head()
df.info()
def label(row):
    if row['label'] < 7:
        return 0 # acid
    else: 
        return 1 # base
df['class'] = df.apply(lambda row: label(row), axis=1)
df.drop('label', axis=1, inplace=True)
df.head()
df['class'].value_counts()
n = df['class'].value_counts()[0] + df['class'].value_counts()[1]
n0 = df['class'].value_counts()[0]
n1 = df['class'].value_counts()[1]

p_0 = n0 / n 
p_1 = n1 / n

print("Probliblity a priori for class\t 0 : {}\t 1: {}".format(round(p_0, 3), round(p_1, 3)))
# acid 
m_blue_acid = df[df['class'] == 0]['blue'].mean()
m_green_acid = df[df['class'] == 0]['green'].mean()
m_red_acid = df[df['class'] == 0]['red'].mean()
# base 
m_blue_base = df[df['class'] == 1]['blue'].mean()
m_green_base = df[df['class'] == 1]['green'].mean()
m_red_base = df[df['class'] == 1]['red'].mean()

# mean acid vector
m_acid = np.array([m_red_acid, m_green_acid, m_blue_acid]).T
m_acid_matrix = np.matrix(m_acid)
# mean base vector
m_base = np.array([m_red_base, m_green_base, m_blue_base]).T
m_base_matrix = np.matrix(m_base)

# only for plotly
m_acid_df = pd.DataFrame(data=m_acid).T
m_acid_df.columns = ['red', 'green', 'blue']
m_base_df = pd.DataFrame(data=m_base).T
m_base_df.columns = ['red', 'green', 'blue']


print("Mean for acid:\n red - {}\n green - {}\n blue - {}".
      format(round(m_red_acid,3), round(m_green_acid,3), round(m_blue_acid,3)))
print("Mean for base:\n red - {}\n green - {}\n blue - {}".
      format(round(m_red_base,3), round(m_green_base,3), round(m_blue_base,3)))
acid_i = np.matrix(df[df['class'] == 0].drop('class', axis=1).values)
base_i = np.matrix(df[df['class'] == 1].drop('class', axis=1).values)
acid_cov_matrix = np.zeros((3, 3))
base_cov_matrix = np.zeros((3, 3))
for i in range(n0):
    acid_cov_matrix += np.dot((acid_i[i].T - m_acid_matrix.T),(acid_i[i] - m_acid_matrix))
for i in range(n1):
    base_cov_matrix += np.dot((base_i[i].T - m_base_matrix.T),(base_i[i] - m_base_matrix))
acid_cov_matrix = acid_cov_matrix / (n0 - 1)
base_cov_matrix = base_cov_matrix / (n1 - 1)
print('Acid covariance matrix: \n{} \nBase covariance matrix: \n{}'.format(acid_cov_matrix, base_cov_matrix))
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
# for acid
cov_acid = df[df['class'] == 0].drop('class', axis=1).cov()
# for base
cov_base = df[df['class'] == 1].drop('class', axis=1).cov()

fig, axn = plt.subplots(1, 2, figsize=(15,5))
sns.heatmap(cov_acid, 
            xticklabels=cov_acid.columns.values,
            yticklabels=cov_acid.columns.values,annot=True,ax=axn[0])
axn[0].set_title('Covariance matrix - acid')
sns.heatmap(cov_base, 
            xticklabels=cov_base.columns.values,
            yticklabels=cov_base.columns.values,annot=True,ax=axn[1])
axn[1].set_title('Covariance matrix - base')
plt.show()
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go


acid = go.Scatter3d(
    x=df[df['class'] == 0]['blue'],
    y=df[df['class'] == 0]['red'],
    z=df[df['class'] == 0]['green'],
    mode='markers',
    marker = dict(size=3,
                  color='rgb(255,0,0)',
                  line=dict(width=0)),
    name ='ACID'
)
base = go.Scatter3d(
    x=df[df['class'] == 1]['blue'],
    y=df[df['class'] == 1]['red'],
    z=df[df['class'] == 1]['green'],
    mode='markers',
    marker = dict(size=3,
                  color='rgb(0,0,255)',
                  line=dict(width=0)),
    name ='BASE'
)

acid_mean = go.Scatter3d(
    x = m_acid_df['blue'],
    y = m_acid_df['green'],
    z = m_acid_df['red'],
    mode='markers',
    marker = dict(size=10,
                  color='rgb(255,20,0)',
                  line=dict(width=3)),
    name = "ACID_MEAN"
)

base_mean = go.Scatter3d(
    x = m_base_df['blue'],
    y = m_base_df['green'],
    z = m_base_df['red'],
    mode='markers',
    marker = dict(size=10,
                  color='rgb(0,20,255)',
                  line=dict(width=3)),
    name = "BASE_MEAN"
)

data = [acid, base, acid_mean, base_mean]
layout = go.Layout(
    title='PH-scale',
    scene = dict(
        xaxis = dict(title='blue'),
        yaxis = dict(title='red'),
        zaxis = dict(title='green'),)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='ph-scale')
W = ((n0 - 1) * acid_cov_matrix + (n1 - 1) * base_cov_matrix) / (n - 2)
W
W_pd = ((n0 - 1) * cov_acid + (n1 - 1) * cov_base) / (n - 2)
W_pd
hat_a = np.linalg.pinv(W)
hat_a = hat_a.dot(m_acid_matrix.T - m_base_matrix.T)
print(hat_a)
b = - 0.5 * hat_a.T
b = b.dot(m_acid_matrix.T - m_base_matrix.T)
print(b)
hat_a_pd = pd.DataFrame(np.linalg.pinv(W_pd.values), W_pd.columns, W_pd.index).values
hat_a_pd = hat_a_pd.dot(m_acid_matrix.T - m_base_matrix.T)
print(hat_a_pd)
b_pd = - 0.5 * hat_a_pd.T
b_pd = b_pd.dot(m_acid_matrix.T - m_base_matrix.T)
print(b_pd)
hyper_d = (0.5 * ((m_base_matrix - m_acid_matrix) * np.linalg.inv(W)) * (m_acid_matrix.T + m_base_matrix.T)).item(0)
hyper_a = ((m_base_matrix - m_acid_matrix) * np.linalg.inv(W)).item(0)
hyper_b = ((m_base_matrix - m_acid_matrix) * np.linalg.inv(W)).item(1)
hyper_c = ((m_base_matrix - m_acid_matrix) * np.linalg.inv(W)).item(2)
print("a : {}\tb : {}\tc : {}\td : {}".format(hyper_a, hyper_b, hyper_c, hyper_d))
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import os

def rotate(angle):
    ax.view_init(azim=angle)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection='3d')

x = np.linspace(0, 250)
y = np.linspace(0, 250)

X,Y = np.meshgrid(x,y)
Z = (-hyper_a*X - hyper_b*Y + hyper_d) * (1. / hyper_c)

# plot the surface
ax.plot_surface(X, Y, Z) # use in animation
ax.scatter(df[df['class'] == 1]['red'],
    df[df['class'] == 1]['green'],
    df[df['class'] == 1]['blue'])
ax.scatter(df[df['class'] == 0]['red'],
    df[df['class'] == 0]['green'],
    df[df['class'] == 0]['blue'])
# plt.axis('off')
# rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,362,2),interval=100)
# rot_animation.save('rotation1.gif', dpi=80, writer='imagemagick')
plt.show()
# naive Bayes 
# LDA or QDA 
# random forest + graphs 
X = np.array(df.drop('class', axis=1))
y = np.array(df['class'])
X.shape, y.shape
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(store_covariance=True)
lda.fit(X, y)
lda_matrix = lda.covariance_
print(lda_matrix)
lda_d = (0.5 * ((m_base_matrix - m_acid_matrix) * np.linalg.inv(lda_matrix)) *(m_acid_matrix.T + m_base_matrix.T)).item(0)
lda_a = ((m_base_matrix - m_acid_matrix) * np.linalg.inv(lda_matrix)).item(0)
lda_b = ((m_base_matrix - m_acid_matrix) * np.linalg.inv(lda_matrix)).item(1)
lda_c = ((m_base_matrix - m_acid_matrix) * np.linalg.inv(lda_matrix)).item(2)
print("a : {}\tb : {}\tc : {}\td : {}".format(lda_a, lda_b, lda_c, lda_d))
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda.fit(X, y)
qda_cov = qda.covariance_ 
qda_cov
acid = go.Scatter3d(
    x=df[df['class'] == 0]['blue'],
    y=df[df['class'] == 0]['red'],
    z=df[df['class'] == 0]['green'],
    mode='markers',
    marker = dict(size=3,
                  color='rgb(255,0,0)',
                  line=dict(width=0)),
    name ='ACID'
)
base = go.Scatter3d(
    x=df[df['class'] == 1]['blue'],
    y=df[df['class'] == 1]['red'],
    z=df[df['class'] == 1]['green'],
    mode='markers',
    marker = dict(size=3,
                  color='rgb(0,0,255)',
                  line=dict(width=0)),
    name ='BASE'
)

x_line = np.linspace(0,200)
y_line = np.linspace(0,200)
X_surface, Y_surface = np.meshgrid(x_line, y_line)
Z = (-hyper_a*X_surface - hyper_b*Y_surface + hyper_d) * (1. / hyper_c)
Z_lda = (-lda_a*X_surface - lda_b*Y_surface + lda_d) * (1. / lda_c)

# something wrong with W matrix
surface = go.Surface(z=X_surface, x=Z, y=Y_surface, opacity=0.9, name ='SURFACE')
surface_lda = go.Surface(z=Y_surface, x=Z_lda, y=X_surface, opacity=0.9, name ='SURFACE_lda')

data_surface = [acid, base, surface, surface_lda]
layout = go.Layout(
    title='Comparation between surfaces',
    scene = dict(
        xaxis = dict(title='blue'),
        yaxis = dict(title='red'),
        zaxis = dict(title='green'),)
)
fig = go.Figure(data=data_surface, layout=layout)
py.iplot(fig,filename='ph-surface')
lda.predict_proba([[255,0,0]]) # only blue -> base 
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
X_reduced[:10]
pca.explained_variance_ratio_
1 - pca.explained_variance_ratio_.sum()
def plot_after_dim_reduce(X_reduced):
    for i in range(len(y)):
        if y[i] == 0:
            plt.scatter(X_reduced[i, 0], X_reduced[i, 1], c='b')
        elif y[i] == 1:
            plt.scatter(X_reduced[i, 0], X_reduced[i, 1], c='r')
    plt.grid(True)
    plt.show()
plot_after_dim_reduce(X_reduced)
from sklearn.decomposition import IncrementalPCA

n_batches = 50
inc_pca = IncrementalPCA(n_components=2)
for X_batch in np.array_split(X, n_batches):
    inc_pca.partial_fit(X_batch)

X_reduced_inc_pca = inc_pca.fit_transform(X)
plot_after_dim_reduce(X_reduced_inc_pca)
X_recoverd_inc_pca = inc_pca.inverse_transform(X_reduced_inc_pca)
print(X_recoverd_inc_pca[:5])
print(X[:5])
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.03)
X_reduced_kernel_rbf = rbf_pca.fit_transform(X)
plot_after_dim_reduce(X_reduced_kernel_rbf)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression())
    ])

param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),
        "kpca__kernel": ["rbf", "sigmoid", "poly", "linear"]
    }]

grid_search = GridSearchCV(clf, param_grid, cv=10)
grid_search.fit(X, y)
print(grid_search.best_params_)
lin_pca = KernelPCA(n_components=2, kernel='linear', gamma=0.03)
X_reduced_kernel_lin = lin_pca.fit_transform(X)
plot_after_dim_reduce(X_reduced_kernel_lin)
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, random_state=42)
X_reduced_lle = lle.fit_transform(X)
plot_after_dim_reduce(X_reduced_lle)
from sklearn.manifold import MDS

mds = MDS(n_components=2, random_state=42)
X_reduced_mds = mds.fit_transform(X)
plot_after_dim_reduce(X_reduced_mds)
from sklearn.manifold import Isomap

isomap = Isomap(n_components=2)
X_reduced_isomap = isomap.fit_transform(X)
plot_after_dim_reduce(X_reduced_isomap)
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42)
X_reduced_tsne = tsne.fit_transform(X)
plot_after_dim_reduce(X_reduced_tsne)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve

y_pred = log_reg.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print('Accuracy : {}'.format(acc))
confusion_matrix(y_test, y_pred)
from sklearn.metrics import roc_auc_score
import scikitplot as skplt

y_pred_proba = log_reg.predict_proba(X_test)
skplt.metrics.plot_precision_recall(y_test, y_pred_proba)
plt.show()
X_train_dr, X_test_dr, y_train_dr, y_test_dr = train_test_split(X_reduced_kernel_lin, y, test_size=0.2)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_dr, y_train_dr)
y_pred_dr = log_reg.predict(X_test_dr)
acc_dr = accuracy_score(y_test_dr, y_pred_dr)
print('Accuracy : {}'.format(acc_dr))
confusion_matrix(y_test_dr, y_pred_dr)
X_train_mds, X_test_mds, y_train_mds, y_test_mds = train_test_split(X_reduced_mds, y, test_size=0.2)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_mds, y_train_mds)
y_pred_mds = log_reg.predict(X_test_mds)
acc_mds = accuracy_score(y_test_mds, y_pred_mds)
print('Accuracy : {}'.format(acc_mds))
confusion_matrix(y_test_mds, y_pred_mds)
X_train_tsne, X_test_tsne, y_train_tsne, y_test_tsne = train_test_split(X_reduced_tsne, y, test_size=0.2)
log_reg = LogisticRegression(random_state=42)
log_reg.fit(X_train_tsne, y_train_tsne)
y_pred_tsne = log_reg.predict(X_test_tsne)
acc_tsne = accuracy_score(y_test_tsne, y_pred_tsne)
print('Accuracy : {}'.format(acc_tsne))
confusion_matrix(y_test_tsne, y_pred_tsne)
