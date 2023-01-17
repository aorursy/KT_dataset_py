import matplotlib.pyplot as plt

import scipy as sc

import numpy as np

from scipy.io import wavfile



fs, data = wavfile.read("../input/voice-example/voz yosef.wav")

data = data/2.0**15





# DATA 2 minutos

training_data = data[0:fs*60]



training_data = training_data.reshape(int(training_data.shape[0]/256),256)
%matplotlib inline



plt.plot(data[fs*0:fs*67])



from IPython.display import Audio

Audio(rate=fs, data=data[fs*0:fs*67])
# Ejemplo de frame

plt.plot(training_data[10])

data2=training_data[10].reshape(-1,1)

Audio(data=training_data[10],rate=fs)
# Construccion del diccionario

from sklearn.decomposition import MiniBatchDictionaryLearning

from time import time

t0 = time()

dico = MiniBatchDictionaryLearning(n_components=1024, alpha=1, n_iter=200)

D = dico.fit(training_data).components_

t1 = time() - t0

print('Tiempo de entrenamiento en %d segundos' % t1)
import itertools

# Ejemplos de diccionario

fig, axes = plt.subplots(6,6)

fig.set_size_inches(16,12)

for i, j in itertools.product(range(1,7), range(6)):

    axes[i-1][j].plot(D[6*(i-1)+j])
# Datos a representar

test_data = data[fs*60:fs*60+111872].reshape(fs*7//256, 256)



# Reconstruccion con 100 atomos

result = np.ndarray((test_data.shape[0],512))



from sklearn.decomposition import SparseCoder



coder = SparseCoder(dictionary = D, transform_n_nonzero_coefs=100, transform_alpha=None, transform_algorithm="omp")



t0 = time()

result = coder.transform(test_data)

t1 = time() - t0

print('Tiempo de entenamiento en %d segundos.' % t1)
orig = data[fs*60:fs*67]

out = np.zeros(orig.shape)



for n in range(result.shape[0]):

    out[n*256:(n+1)*256] = np.sum(D.T*result[n],axis=1)
#Audio reconstruido

Audio(data=out,rate=fs)
fig, axes = plt.subplots(3)

fig.set_size_inches(10,8)



axes[0].plot(orig)

axes[1].plot(out, 'g')

axes[2].plot((out-orig)**2, 'r')
frame=0

idx_r, = result[frame].nonzero()

plt.xlim(0, 1024)

plt.title("Primer frame de sonido")

plt.stem(idx_r, result[frame][idx_r], use_line_collection=True)

plt.suptitle('Reconstrucción de señal con OMP',

             fontsize=16)

plt.show()