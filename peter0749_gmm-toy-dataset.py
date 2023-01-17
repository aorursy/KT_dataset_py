import numpy as np

from sklearn.mixture import GaussianMixture

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats

from matplotlib.colors import LogNorm

def data_generator(cluster_nums, cluster_means, cluster_var, bg_range, bg_noise_n):

    data = []

    for num, mu, var in zip(cluster_nums, cluster_means, cluster_var):

        data += [np.random.multivariate_normal(mu, np.diag(var), num)]

    data = np.vstack(data)

    noise = np.random.uniform(bg_range[0], bg_range[1], size=(bg_noise_n, data.shape[-1]))

    data = np.append(data, noise, axis=0)

    return data
batch_data = data_generator([400,600,800], [[0.5, 0.5], [6, 1.5], [1, 7]], [[1, 3], [2, 2], [6, 2]], [[-10, -15],[15, 20]], 30)

print(batch_data.shape)
plt.scatter(batch_data[...,0], batch_data[...,1])

plt.show()
gmm = GaussianMixture(n_components=3, tol=1e-3, max_iter=300, covariance_type='diag')
gmm.fit(batch_data)
means = np.array(gmm.means_)

vars_ = np.array(gmm.covariances_)

stds = np.sqrt(vars_)

print('means: ', means)

print('stds: ', stds)
def det_outer(x, label, means, stds, percentile=0.9):

    # 使用 percentile 回推幾倍 stdev ，超過那個數值代表 outlier

    dev = scipy.stats.norm.ppf(percentile)

    #print(dev)

    y = (x - means[label]) / stds[label]

    y = np.sqrt((y**2).sum(axis=-1)) > dev

    return y
cluster_label = np.asarray(gmm.predict(batch_data))
# abnormal_pts = det_outer(batch_data, cluster_label, means, stds, 0.997) # label 0

pts_score = gmm.score_samples(batch_data)

log_likelihood_threshold = np.percentile(pts_score, 1) # < 1%

log_likelihood_threshold
sns.distplot(pts_score, hist = False, kde = True,

                 kde_kws = {'linewidth': 3},

                 label = 'Log likelyhood')

plt.axvline(log_likelihood_threshold, color='red')

plt.title('Dense plot of log likelyhood')

plt.show()
abnormal_pts = pts_score < log_likelihood_threshold

cluster_label += 1

cluster_label[abnormal_pts] = 0
x = np.linspace(batch_data[...,0].min(), batch_data[...,0].max())

y = np.linspace(batch_data[...,1].min(), batch_data[...,1].max())

X, Y = np.meshgrid(x, y)

XX = np.array([X.ravel(), Y.ravel()]).T

Z = -gmm.score_samples(XX)

Z = Z.reshape(X.shape)
def plot_label(x_emb,y,n,means,X,Y,Z,Z_levels,title=''):

    means_c = np.array(means)

    cmap = plt.cm.gist_ncar_r

    fig, ax = plt.subplots()

    for l in range(n):

        points = x_emb[y==l,:]

        ax.scatter(points[:,0], points[:,1], label=l, c=cmap(float(l+1)/float(n+1)))

    ax.legend(loc='lower right', frameon=True, prop={'size': 10})

    ax.set_title(title)

    ax.scatter(means_c[:,0], means_c[:,1], marker='x')

    cs = ax.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=60.0), levels=Z_levels, cmap=plt.cm.get_cmap('Blues_r'))

    norm= matplotlib.colors.Normalize(vmin=cs.cvalues.min(), vmax=cs.cvalues.max())

    sm = plt.cm.ScalarMappable(norm=norm, cmap = cs.cmap)

    sm.set_array([])

    fig.colorbar(sm, ticks=cs.levels)

    plt.show()
plot_label(batch_data,cluster_label,4,means,X,Y,Z,np.logspace(0, 3, 30),title='Clustering')