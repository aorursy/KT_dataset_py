import numpy as np

from scipy.fft import fft, fftshift, ifft, ifftshift

import plotly.express as xp

import plotly.graph_objects as go



import matplotlib.pyplot as plt
from compressed_sensing_functions_py import plot_nyquist_demo



plot_nyquist_demo()
l = 128

n = 5

sigma = 0.05

np.random.seed(42)



#generate sparse signal

x = np.concatenate( (np.ones(n) / n , np.zeros(l-n)) , axis=0 )

x = np.random.permutation(x)

# add random noise

y = x + sigma * np.random.randn(l)



fig = go.Figure()

fig.add_trace(

    go.Scatter( x=  np.arange(l) , y = x , name='x')

)

fig.add_trace(

    go.Scatter( x=  np.arange(l) , y = y , name='y')

)
fig = go.Figure()

for lam in [0.01,0.05, 0.1, 0.2]:

    fig.add_trace(

        go.Scatter( x=  np.arange(l) , y = 1/(1+lam) * y , name=f"lambda = {str(lam)}")

    )

fig.show()
def soft_thresh(x, lam):

    if ~(isinstance(x[0], complex)):

        return np.zeros(x.shape) + (x + lam) * (x<-lam) + (x - lam) * (x>lam) 

    else:

        return np.zeros(x.shape) + ( abs(x) - lam ) / abs(x) * x * (abs(x)>lam) 


fig = go.Figure()

for i, lam in enumerate([0.01,0.05, 0.1, 0.2]):

    fig.add_trace(

        go.Scatter( x=  np.arange(l) , y = soft_thresh(y, lam)+i/10 , name=f"lambda = {str(lam)}")

    )

fig.show()
def fftc(x):

    """Computes the centered Fourier transform"""

    return fftshift( fft(x) )



def ifftc(X):

    """Inverses the centered Fourier transform"""

    return ifft( ifftshift(X) )
X = fftc(x)

Y = fftc(y)

fig = go.Figure()

fig.add_trace( go.Scatter(x = np.arange(-l/2,l-1),y=abs(X) , name='X') )

fig.add_trace( go.Scatter(x = np.arange(-l/2,l-1),y=abs(Y) , name='Y') )

fig.show()
#uniformly sampled k-space

Xu = 4 * X

for i in range(1,4):

    Xu[i::4] = 0

#reconstructed signal

xu = ifftc(Xu)



#randomly sampled k-space

Xr = 4 * X * np.random.permutation(np.repeat([1,0,0,0], l/4) )

xr = ifftc( Xr )



#plot the comparison

fig = go.Figure()

fig.add_trace( go.Scatter(y=x*1.5, name='original signal (scaled)'))

fig.add_trace( go.Scatter(y=xu.real, name='uniform sampling'))

fig.add_trace( go.Scatter(y=xr.real, name='random sampling'))
# undersampled noisy signal in k-space and let this be first order Xhat

Y = 4 * fftc(x) * np.random.permutation(np.repeat([1,0,0,0], l/4) )

Xhat = Y.copy()





# Repeat steps 1-4 until change is below a threshold

eps = 1e-4

lam = 0.05



def distance(x,y):

    return abs(sum(x-y))

diff=[]

err = []

itermax = 10000

while True:

    itermax -= 1

    xhat_old = ifftc(Xhat)

    xhat = soft_thresh(xhat_old, lam)

    diff.append(distance(xhat, xhat_old))

    err.append(distance(xhat.real/4,x))

    if ( diff[-1] < eps ) | ( itermax == 0 ):

        break

    Xhat = fftc(xhat)

    Xhat[Y!=0] = Y[Y!=0]

    

    

fig = go.Figure()

fig.add_trace( go.Scatter( y = x , name = 'true signal') )

fig.add_trace( go.Scatter( y = ifftc(Y).real, name = 'reconstruction before noise reduction'))

fig.add_trace( go.Scatter( y = xhat.real/4, name = 'reconstructed after noise reduction'))

fig = go.Figure()

fig.add_trace( go.Scatter( y = err) )

fig.update_layout( title = 'Error at each step' )

fig.show()
fig = go.Figure()

fig.add_trace( go.Scatter( y = diff) )

fig.update_layout( title = 'Differential error at each step' )

fig.show()
import pywt

import pywt.data
# Load image

original = pywt.data.camera()



# Wavelet transform of image, and plot approximation and details

titles = ['Approximation', ' Horizontal detail',

          'Vertical detail', 'Diagonal detail']

coeffs2 = pywt.dwt2(original, 'bior1.3')

LL, (LH, HL, HH) = coeffs2

fig = plt.figure(figsize=(10, 10*4))

for i, a in enumerate([LL, LH, HL, HH]):

    ax = fig.add_subplot(4, 1, i + 1)

    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)

    ax.set_title(titles[i], fontsize=10)

    ax.set_xticks([])

    ax.set_yticks([])



fig.tight_layout()

plt.show()
undersample_rate = .5

n = original.shape[0] * original.shape[1]

original_undersampled = ( original.reshape(-1) \

    * np.random.permutation( 

        np.concatenate( 

            (np.ones( int( n * undersample_rate ) ), 

             np.zeros( int( n * ( 1-undersample_rate )) )) 

        ) 

    ) 

                        ).reshape(512,512)
fig,ax = plt.subplots(1,2,figsize=(20,10))

ax[0].imshow(original, cmap=plt.cm.gray)

ax[1].imshow(original_undersampled,cmap=plt.cm.gray)
def flat_wavelet_transform2(x, method='bior1.3'):

    """For a 2D image x, take the wavelet """

    coeffs = pywt.wavedec2( x, method )

    output = coeffs[0].reshape(-1)

    for tups in coeffs[1:]:

        for c in tups:

            output = np.concatenate((output, c.reshape(-1)))

    return output



def inverse_flat_wavelet_transform2(X,  shape, method='bior1.3'):

    shapes = pywt.wavedecn_shapes( shape , method)

    nx = shapes[0][0]

    ny = shapes[0][1]

    n = nx * ny

    coeffs = [X[:n].reshape(nx,ny) ]

    for i, d in enumerate(shapes[1:]):

        vals=list(d.values())

        nx = vals[0][0]

        ny = vals[0][1]

        coeffs.append( (X[ n : n + nx * ny].reshape( nx, ny ), 

                        X[ n + nx * ny : n + 2 * nx * ny ].reshape( nx, ny ), 

                        X[ n + 2 * nx * ny : n + 3 * nx * ny ].reshape( nx, ny ))  )

        n += 3 * nx * ny

    return pywt.waverec2(coeffs, method)
plt.figure(figsize=(15,10))

plt.plot(flat_wavelet_transform2(original_undersampled, 'bior1.3')[-100:] )

plt.title('A small sample of the wavelet transform')
plt.figure(figsize=(15,10))

plt.hist(flat_wavelet_transform2(original_undersampled, 'bior1.3') , range=(-300,300), bins=30)

plt.title('A small sample of the wavelet transform')
methods = ['haar','coif1','coif2','coif3','bior1.1','bior1.3','bior3.1','bior3.3','rbio1.1','rbio1.3','rbio3.1','rbio3.3']

def distance(x,y):

    return sum(abs(x.reshape(-1)-y.reshape(-1)))



# undersampled noisy signal in image-space and let this be first order Xhat

y = original_undersampled



# Repeat steps 1-4 until change is below a threshold

eps = 1e-2

lam = 100

lam_decay = 0.995

minlam = 1



err2=[]





lam = 100





xhat = y.copy()

for i in range(80):

    method = 'sym3'

    xhat_old = xhat

    Xhat_old = flat_wavelet_transform2(xhat, method)

    Xhat = soft_thresh(Xhat_old, lam)

    xhat = inverse_flat_wavelet_transform2(Xhat, (512,512), method)

    xhat[y!=0] = y[y!=0]   





    xhat = xhat.astype(int)

    xhat[xhat<0] = 0

    xhat[xhat>255] = 255

    err2.append(distance(original, xhat))

    lam *= lam_decay 

#     if (distance(xhat, xhat_old)<eps):

#         break







    

fig = plt.figure(figsize=(10,10))  

plt.loglog(err2)





fig,ax = plt.subplots(1,2,figsize=(20,10))

ax[0].imshow(original, cmap=plt.cm.gray)

ax[1].imshow(xhat,cmap=plt.cm.gray, vmin=0, vmax=255)