import matplotlib.pyplot as plt
import scipy as sc
import numpy as np
from scipy.io import wavfile

#Se extrae los datos y la frecuenci de muestreo 
fs, data = wavfile.read("../input/voice-example/voz yosef.wav")
data = data/2.0**15
#fs es igual a 100 frames de 160 muestreos cada uno.
data=data[0:fs*67]
# Data de entrenamiento (1 min =60 Frame Samples(fs))
training_data = data[0:fs*60]
training_data = training_data.reshape(6000,160)
%matplotlib inline

plt.plot(data[fs*0:fs*67])

from IPython.display import Audio
Audio(rate=fs, data=data[fs*0:fs*67])
# Ejemplo de frame
plt.plot(training_data[10])
data2=training_data[10].reshape(-1,1)
Audio(data=training_data[10],rate=fs)
import numpy as np
import scipy as sp
from sklearn.linear_model import orthogonal_mp_gram, OrthogonalMatchingPursuit


class KSVD(object):
    def __init__(self, n_components, max_iter=10, tol=1e-6,
                 transform_n_nonzero_coefs=None):
        """
        Parameters
        ----------
        n_components:
            Number of dictionary elements
        max_iter:
            Maximum number of iterations
        tol:
            tolerance for error
        transform_n_nonzero_coefs:
            Number of nonzero coefficients to target
        """
        self.components_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

    def _update_dict(self, Y, D, X):
        for j in range(self.n_components):
            I = X[:, j] > 0
            if np.sum(I) == 0:
                continue

            D[j, :] = 0
            g = X[I, j].T
            r = Y[I, :] - X[I, :].dot(D)
            d = r.T.dot(g)
            d /= np.linalg.norm(d)
            g = r.dot(d)
            D[j, :] = d
            X[I, j] = g.T
        return D, X

    def _initialize(self, Y):
        if min(Y.shape) < self.n_components:
            D = np.random.randn(self.n_components, Y.shape[1])
        else:
            u, s, vt = sp.sparse.linalg.svds(Y, k=self.n_components)
            D = np.dot(np.diag(s), vt)
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
        return D

    def _transform(self, D, Y):
        gram = D.dot(D.T)
        Xy = D.dot(Y.T)
        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * Y.shape[1])
        return orthogonal_mp_gram(
            gram, Xy, n_nonzero_coefs=n_nonzero_coefs).T

    def fit(self, Y):
        """
        Parameters
        ----------
        Y: shape = [n_samples, n_features]
        """
        D = self._initialize(Y)
        for i in range(self.max_iter):
            X = self._transform(D, Y)
            e = np.linalg.norm(Y - X.dot(D))
            if e < self.tol:
                break
            D, X = self._update_dict(Y, D, X)

        self.components_ = D
        return self

    def transform(self, Y):
        return self._transform(self.components_, Y)
# Y ~ X.dot(D)
aksvd = KSVD(n_components=640,transform_n_nonzero_coefs=10)
D = aksvd.fit(training_data).components_
print(D.shape)
import itertools
# Ejemplos de diccionario
fig, axes = plt.subplots(6,6)
fig.set_size_inches(16,12)
for i, j in itertools.product(range(1,7), range(6)):
    axes[i-1][j].plot(D[6*(i-1)+j])

# Datos a representar
test_data = data[fs*60:fs*67].reshape(700, 160)

# Reconstruccion con 100 atomos

from sklearn.decomposition import SparseCoder

coder = SparseCoder(dictionary = D, transform_n_nonzero_coefs=10, transform_alpha=None, transform_algorithm="omp")


result = coder.transform(test_data)

plt.plot(result[436])
orig = data[fs*60:fs*67]
out = np.zeros(orig.shape)

for n in range(result.shape[0]):
    out[n*160:(n+1)*160] = np.sum(D.T*result[n],axis=1)
plt.plot(out, 'g')
Audio(data=out,rate=fs)
fig, axes = plt.subplots(3)
fig.set_size_inches(10,8)

axes[0].plot(orig)
axes[1].plot(out, 'g')
axes[2].plot((out-orig)**2, 'r')
frame=0
idx_r, = result[frame].nonzero()
plt.xlim(0, 600)
plt.title("Primer frame de sonido")
plt.stem(idx_r, result[frame][idx_r], use_line_collection=True)
plt.suptitle('Reconstrucción de señal con OMP',
             fontsize=16)
plt.show()
Residuals=[]
for i in range(100):
  test_data = data[fs*60:fs*60+160].reshape(1, 160)
  result = np.ndarray((test_data.shape[0],512))
  from sklearn.decomposition import SparseCoder
  coder = SparseCoder(dictionary = D, transform_n_nonzero_coefs=i+1, transform_alpha=None, transform_algorithm="omp")
  result = coder.transform(test_data)
  orig = data[fs*60:fs*60+160]
  out = np.zeros(orig.shape)
  for n in range(result.shape[0]):
      out[n*160:(n+1)*160] = np.sum(D.T*result[n],axis=1)
  Residuals.append(np.sum((test_data-out)**2)**(1/2))
print(Residuals)
plt.plot(np.arange(160),out,np.arange(160),test_data.reshape(160))
plt.plot(np.arange(100),Residuals)