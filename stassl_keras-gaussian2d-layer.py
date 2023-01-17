import math

import numpy as np



from keras.layers import Input

from keras import backend as K

from keras.engine.topology import Layer



from skimage.util.montage import montage2d
def nbimage( data, vmin = None, vmax = None, vsym = False, saveas = None ):

    '''

    Display raw data as a notebook inline image.



    Parameters:

    data: array-like object, two or three dimensions. If three dimensional,

          first or last dimension must have length 3 or 4 and will be

          interpreted as color (RGB or RGBA).

    vmin, vmax, vsym: refer to rerange()

    saveas: Save image file to disk (optional). Proper file name extension

            will be appended to the pathname given. [ None ]

    '''

    from IPython.display import display, Image

    from PIL.Image import fromarray

    from io import BytesIO

    data = rerange( data, vmin, vmax, vsym )

    data = data.squeeze()

    # try to be smart

    if data.ndim == 3 and 3 <= data.shape[ 0 ] <= 4:

        data = data.transpose( ( 1, 2, 0 ) )

    s = BytesIO()

    fromarray( data ).save( s, 'png' )

    if saveas is not None:

        open( saveas + '.png', 'wb' ).write( s )

    display( Image( s.getvalue() ) )

    

def rerange( data, vmin = None, vmax = None, vsym = False ):

    '''

    Rescale values of data array to fit the range 0 ... 255 and convert to uint8.



    Parameters:

    data: array-like object. if data.dtype == uint8, no scaling will occur.

    vmin: original array value that will map to 0 in the output. [ data.min() ]

    vmax: original array value that will map to 255 in the output. [ data.max() ]

    vsym: ensure that 0 will map to gray (if True, may override either vmin or vmax

          to accommodate all values.) [ False ]

    '''

    from numpy import asarray, uint8, clip

    data = asarray( data )

    if data.dtype != uint8:

        if vmin is None:

            vmin = data.min()

        if vmax is None:

            vmax = data.max()

        if vsym:

            vmax = max( abs( vmin ), abs( vmax ) )

            vmin = -vmax

        data = ( data - vmin ) * ( 256 / ( vmax - vmin ) )

        data = clip( data, 0, 255 ).astype( uint8 )

    return data
class Gaussian2D(Layer):

    def __init__(self, output_shape, **kwargs):

        self.output_shape_ = output_shape

        self.height = output_shape[2]

        self.width = output_shape[3]

        self.grid = np.dstack(np.mgrid[-1:1:(2. / self.height), -1:1:(2. / self.width)])[None, ...]

        super(Gaussian2D, self).__init__(**kwargs)



    def call(self, inputs, mask=None):

        mu, sigma, corr, scale = inputs

        mu = K.tanh(mu) * 0.95

        sigma = K.exp(sigma) + 0.00001

        corr = K.tanh(corr[:, 0]) * 0.95

        scale = K.exp(scale[:, 0])



        mu0 = K.permute_dimensions(mu[:, 0], (0, 'x', 'x', 'x'))

        mu1 = K.permute_dimensions(mu[:, 1], (0, 'x', 'x', 'x'))

        sigma0 = K.permute_dimensions(sigma[:, 0], (0, 'x', 'x', 'x'))

        sigma1 = K.permute_dimensions(sigma[:, 1], (0, 'x', 'x', 'x'))

        grid0 = self.grid[..., 0]

        grid1 = self.grid[..., 1]

        corr = K.permute_dimensions(corr, (0, 'x', 'x', 'x'))

        scale = K.permute_dimensions(scale, (0, 'x', 'x', 'x'))

        

        return K.tanh(scale / (2. * math.pi * sigma0 * sigma1 * K.sqrt(1. - corr * corr)) *

              K.exp(-(1. / (2. * (1. - corr * corr)) *

                      ((grid0 - mu0) * (grid0 - mu0) / (sigma0 * sigma0) +

                       (grid1 - mu1) * (grid1 - mu1) / (sigma1 * sigma1) -

                       2. * corr * (grid0 - mu0) * (grid1 - mu1) / sigma0 / sigma1))))



    def get_output_shape_for(self, input_shape):

        return self.output_shape_
mu_input = Input((2,))

sigma_input = Input((2,))

corr_input = Input((1,))

scale_input = Input((1,))

g = Gaussian2D(output_shape=(None, 1, 100, 100))([mu_input, sigma_input, corr_input, scale_input])



n = 3*3

mu = np.random.normal(size=(n,2))/3

sigma = np.random.uniform(-3, -2, size=(n,2))

corr = np.random.normal(size=(n,1))/5

scale = np.random.normal(size=(n,1))



gaussians = g.eval({mu_input: mu.astype('float32'), 

        sigma_input: sigma.astype('float32'), 

        corr_input: corr.astype('float32'), 

        scale_input: scale.astype('float32')})



nbimage(montage2d(gaussians.squeeze().clip(0, 1)))