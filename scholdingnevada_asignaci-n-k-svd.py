import matplotlib.pyplot as plt
import scipy as sc
import numpy as np
from scipy.io import wavfile

#Se extrae los datos y la frecuenci de muestreo 
fs, data = wavfile.read("../input/voice-example/voz yosef.wav")
data = data/2.0**15
#fs es igual a 100 frames de 160 muestreos cada uno.
data=data[0:fs*67]
# Data de entrenamiento (20 seg =20 Frame Samples(fs))
training_data = data[0:fs*20]
training_data = training_data.reshape(2000,160)
%matplotlib inline

plt.plot(data[fs*0:fs*67])

from IPython.display import Audio
Audio(rate=fs, data=data[fs*0:fs*67])
# Ejemplo de frame
plt.plot(training_data[2])
data2=training_data[2].reshape(-1,1)
Audio(data=training_data[2],rate=fs)
plt.plot(training_data[14])
plt.plot(training_data[15])
plt.plot(training_data[16])
plt.title("Señales $y_i$ de $Y$")
plt.legend(["$y_{14}$", "$y_{15}$","$y_{16}$"])
import numpy as np
import scipy as sp
from sklearn.linear_model import orthogonal_mp_gram, OrthogonalMatchingPursuit

T=0
Error=[]
Error2=[]
Error_n_K=[]
class KSVD(object):
    def __init__(self, n_components, max_iter=10, tol=1e-6,transform_n_nonzero_coefs=None):
        self.components_ = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs

    def _update_dict(self, Y, D, X):
        #Actualización del diccionario por medio de svd de E
        Error3=[]
        for j in range(self.n_components):
            I = X[:, j] != 0
            if np.sum(I) == 0:
                continue
            D[j, :] = 0
            g = X[I, j].T
            #calculo del error
            r = Y[I, :] - X[I, :].dot(D)
            #descomposición por SVD del error
            u,s,vh = np.linalg.svd(r.T,full_matrices=True)
            delta = s[0]
            #actualización del atomo d_j
            D[j, :] = u[:,0]
            #actualización de la representación x_j
            X[I, j] = delta*np.array(vh[0,:])
            
            #calculo de error total
            Y_prima_k= D.T.dot(X.T)
            Error_k= np.sum((Y.T-Y_prima_k)**2,axis=0)
            Error_k=np.sum(Error_k)
            Error3.append(Error_k)
            Error_n_K.append(Error_k)
            
        return D, X

    def _initialize(self, Y):
        #Selección de D_0
        if min(Y.shape) < self.n_components:
            D = np.random.randn(self.n_components, Y.shape[1])
        else:
            u, s, vt = sp.sparse.linalg.svds(Y, k=self.n_components)
            D = np.dot(np.diag(s), vt)
        #Normalización de los datos
        D /= np.linalg.norm(D, axis=1)[:, np.newaxis]
        return D

    def _transform(self, D, Y):
        #Aplicación de OMP para creacion y optimización de X
        gram = D.dot(D.T)
        Xy = D.dot(Y.T)
        n_nonzero_coefs = self.transform_n_nonzero_coefs
        if n_nonzero_coefs is None:
            n_nonzero_coefs = int(0.1 * Y.shape[1])
        return orthogonal_mp_gram(
            gram, Xy, n_nonzero_coefs=n_nonzero_coefs).T
    

    def fit(self, Y):
        D = self._initialize(Y)
        for i in range(self.max_iter):
            T=i
            X = self._transform(D, Y)
            e = np.linalg.norm(Y - X.dot(D))
            if e < self.tol:
                break
            D, X = self._update_dict(Y, D, X)
            out= np.sum(D.T*X[0,:],axis=1)
            Error.append(np.sum((out-Y[0,:])**2))
            Y_prima= D.T.dot(X.T)
            Error_N= np.sum((Y.T-Y_prima)**2,axis=0)
            Error_N=np.sum(Error_N)
            Error2.append(Error_N)  
        self.components_ = D,X
        return self

    def transform(self, Y):
        return self._transform(self.components_, Y)
# Y ~ D@X
ksvd = KSVD(n_components=640,transform_n_nonzero_coefs=10)
D,X = ksvd.fit(training_data).components_

fig, axes = plt.subplots(2,5)
fig.set_size_inches(16,12)
for i, j in itertools.product(range(0,2), range(5)):
    axes[i][j].plot(Error_n_K[(((4*i)+j)*640):(((4*i)+j+1)*640)])
plt.plot(np.arange(160),training_data[14],np.arange(160),D.T@X.T[:,14])
plt.plot(np.arange(160),training_data[15],np.arange(160),D.T@X.T[:,15])
plt.plot(np.arange(160),training_data[16],np.arange(160),D.T@X.T[:,16])
import itertools
# Ejemplos de diccionario
fig, axes = plt.subplots(6,6)
fig.set_size_inches(16,12)
for i, j in itertools.product(range(1,7), range(6)):
    axes[i-1][j].plot(D[6*(i-1)+j])