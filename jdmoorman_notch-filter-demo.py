import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import imageio
from skimage.transform import resize

# globals go here
figsize = (6,6)
f = imageio.imread("../input/Fig5.26a.jpg").astype(float)
f = resize(f, (100,100), mode='edge')
N, M = f.shape

plt.imshow(f, cmap='gray')
plt.show()
def centered_fft(f): 
    """
    fftshift(fft(f))
    """
    N, M = f.shape
    
    # multiply f by (-1)^(x+y) to shift the center 
    f = np.copy(f)
    for i in range(N):
        for j in range(M):
            d = (i)+(j)
            f[i,j] *= (-1)**d

    # compute the DFT of f*(-1)^{x+y} 
    return np.fft.fft2(f)
def log_abs(f):
    """
    Use for plotting fft spectra
    """
    return np.log(1 + np.abs(f))
def centered_ifft(F):
    """
    ifft(fftshift(F))
    """
    
    N, M = F.shape

    # apply IFFT and take the real part 
    f = np.real(np.fft.ifft2(F));

    # multiply f by (-1)^(x+y) to shift the center back to the origin 
    for i in range(M):
        for j in range(N):
            d = (i)+(j);
            f[i,j] *= (-1)**d;
    
    return f
def sin_wave(M, N, u0, v0, phase):
    """
     .-.     .-.     .-.
    /   \   /   \   /   \   /
         `-´     `-´     `-´
    """
    
    x, y = np.meshgrid(np.arange(M), np.arange(N))

    return np.sin(2*np.pi*(u0*x/M + v0*y/N + phase))
def square_wave(M, N, u0, v0, phase, thresh):
    """
       |¯¯¯|   |¯¯¯|   |¯¯¯
    ___|   |___|   |___|
    """
    noise = sin_wave(M, N, u0, v0, phase)
    noise[noise>thresh] = 1
    noise[noise==thresh] = 0
    noise[noise<thresh] = -1
    return noise
def triangle_wave(M, N, u0, v0, phase):
    """
     /\    /\    /\    /\
    /  \  /  \  /  \  /  \  /
        \/    \/    \/    \/
    """
    noise = sin_wave(M, N, u0, v0, phase)
    return np.arcsin(noise)
"""
Here we add periodic noise to an image
"""

# These are parameters
u0 = 3
v0 = 2
phase = 0
thresh = 0 # only used in square wave, takes values in (-1, 1)


# noise = sin_wave(M, N, u0, v0, phase)
# noise = square_wave(M, N, u0, v0, phase, thresh)
noise = triangle_wave(M, N, u0, v0, phase)

# add noise to image
g = noise
# g = 255*noise + f

# show the noisy image
plt.figure(figsize=figsize)
plt.title("u0 = {}, v0 = {}, phase = {} image".format(u0, v0, phase))
plt.imshow(g, cmap='gray')
plt.colorbar()

# wireframe of the noisy image
fig = plt.figure(figsize=figsize)
ax = Axes3D(fig)
x, y = np.meshgrid(np.arange(M), np.arange(N))
ax.plot_wireframe(x, y, g)

G = centered_fft(g)

# image spectrum
plt.figure(figsize=figsize)
plt.title("u0 = {}, v0 = {}, phase = {} log spectrum".format(u0, v0, phase))
plt.imshow(log_abs(G), cmap='gray')
plt.colorbar()

plt.show()
"""
The goal here is to filter the periodic noise out of G using a notch filter
"""

H = np.copy(G)

# This is basically a notch filter on carefully chosen frequencies
H[round(N/2-v0), round(M/2-u0)] = 0
H[round(N/2+v0), round(M/2+u0)] = 0

# # Optionally remove some more carefully chosen frequencies
# for k in range(8):
#     H[round(N/2-(2*k+1)*v0), round(M/2-(2*k+1)*u0)] = 0
#     H[round(N/2+(2*k+1)*v0), round(M/2+(2*k+1)*u0)] = 0
    
plt.figure(figsize=figsize)
plt.title("u0 = {}, v0 = {}, phase = {} log spectrum".format(u0, v0, phase))
plt.imshow(log_abs(H), cmap='gray')
plt.colorbar()

h = centered_ifft(H)

plt.figure(figsize=figsize)
plt.title("u0 = {}, v0 = {}, phase = {} image".format(u0, v0, phase))
plt.imshow(h, cmap='gray')
plt.colorbar()

plt.show()