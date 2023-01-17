import numpy as np
def dft(x):

    x = np.asarray(x)

    N = x.shape[0]

    n = np.arange(N)

    k = np.reshape(n, (N, 1))

    M = np.exp(-2j*np.pi/N*k*n)

    return np.dot(M, x)
x = np.random.random(1024)

np.allclose(dft(x), np.fft.fft(x))
def fft(x):

    N = x.shape[0]

    if N <= 2:

        return dft(x)

    else:

        x_even = fft(x[::2])

        x_odd = fft(x[1::2])

        terms = np.exp(-2j * np.pi * np.arange(N) / N)

        return np.concatenate([x_even + terms[:int(N/2)]*x_odd,

                               x_even + terms[int(N/2):]*x_odd])
np.allclose(fft(x), np.fft.fft(x))
def fft_v(x):

    N = x.shape[0]

    N_min = min(N, 2)

    n = np.arange(N_min)

    k = n[:, None]

    terms = np.exp(-2j * np.pi * k * n / N_min)

    X = np.dot(terms, x.reshape(N_min, -1))

    

    while X.shape[0] < N:

        X_even = X[:, :int(X.shape[1] / 2)]

        X_odd = X[:, int(X.shape[1] / 2):]

        terms = np.exp(-1j * np.pi * np.arange(X.shape[0]) / X.shape[0])[:, None]

        X = np.vstack([X_even + terms * X_odd,

                       X_even - terms * X_odd])

    return X.ravel()
np.allclose(fft_v(x), np.fft.fft(x))
%timeit np.fft.fft(x)

%timeit dft(x)

%timeit fft(x)

%timeit fft_v(x)