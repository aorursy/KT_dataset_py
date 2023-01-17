import numpy as np

from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from mpl_toolkits.mplot3d import proj3d

from matplotlib.patches import FancyArrowPatch
np.random.seed(0) # random seed for consistency



mu_vec1 = np.array([0,0,0])

cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])

class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 40).T



mu_vec2 = np.array([1,2,3])

cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])

class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 40).T



all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')

ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'go', markersize=8, alpha=0.5, label='Class 1')

ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], 'b*', markersize=8, alpha=0.5, label='Class 2')

plt.legend()
mean_x = np.mean(all_samples[0,:])

mean_y = np.mean(all_samples[1,:])

mean_z = np.mean(all_samples[2,:])



mean_vector = np.array([[mean_x],[mean_y],[mean_z]])



print('Mean Vector:\n', mean_vector)
all_samples_meaned = all_samples - mean_vector

cov_mat = 1/len(all_samples_meaned)*all_samples_meaned.dot(all_samples_meaned.T)

print('Covariance Matrix:\n', cov_mat)
# eigenvectors and eigenvalues from the covariance matrix using eigendecomposition

eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)



for i in range(len(eig_val_cov)):

    eigvec_cov = eig_vec_cov[:,i].reshape(1,3).T



    print('Eigenvector {}: \n{}'.format(i+1, eigvec_cov))

    print('Eigenvalue {}: {}'.format(i+1, eig_val_cov[i]))

    print(40 * '-')
eig_vec_cov.dot(np.identity(3)*eig_val_cov).dot(eig_vec_cov.T)
class Arrow3D(FancyArrowPatch):

    def __init__(self, xs, ys, zs, *args, **kwargs):

        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)

        self._verts3d = xs, ys, zs



    def draw(self, renderer):

        xs3d, ys3d, zs3d = self._verts3d

        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)

        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))

        FancyArrowPatch.draw(self, renderer)
fig = plt.figure(figsize=(10,10))

ax = fig.add_subplot(111, projection='3d')

ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'go', markersize=8, alpha=0.5, label='Class 1')

ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], 'b*', markersize=8, alpha=0.5, label='Class 2')



for v in eig_vec_cov.T:

    a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")

    ax.add_artist(a)
# Sort the (eigenvalue, eigenvector) from high to low

idx = np.argsort(eig_val_cov)[::-1]

eig_val_cov = eig_val_cov[idx]

eig_vec_cov = eig_vec_cov[idx]



# Visually confirm that the list is correctly sorted by decreasing eigenvalues

for val, vec in zip(eig_val_cov, eig_vec_cov):

    print(val, '\t', vec)
matrix_w = eig_vec_cov[:,:2]

print('Matrix W:\n', matrix_w)
transformed = matrix_w.T.dot(all_samples)
plt.plot(transformed[0,:40], transformed[1,:40], 'go', alpha=0.5, label='class1')

plt.plot(transformed[0,40:], transformed[1,40:], 'b*', alpha=0.5, label='class2')

plt.legend();

plt.title('Transformed samples with class labels');