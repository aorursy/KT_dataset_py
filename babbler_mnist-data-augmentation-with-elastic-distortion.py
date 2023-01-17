import numpy as np

from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

def elastic_transform(image, alpha_range, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
       
   # Arguments
       image: Numpy array with shape (height, width, channels). 
       alpha_range: Float for fixed value or [lower, upper] for random value from uniform distribution.
           Controls intensity of deformation.
       sigma: Float, sigma of gaussian filter that smooths the displacement fields.
       random_state: `numpy.random.RandomState` object for generating displacement fields.
    """
    
    if random_state is None:
        random_state = np.random.RandomState(None)
        
    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)
%matplotlib inline

import keras
import matplotlib.pyplot as plt
x_train = np.loadtxt('../input/train.csv', dtype=int, delimiter=',', skiprows=1)
x_train = np.reshape(x_train[:, 1:], (42000, 28, 28))
def plot_digits(examples, title=None, size_mult=1):
    """Intended for graphing MNIST digits. 
    
    # Arguments
        examples: Numpy array with shape (num_examples, height, width, num_iterations).
        title: Plot title string.
        size_mult: Multiply figsize by `size_mult`.
    """
   
    num_iterations = examples.shape[-1]
    num_examples = examples.shape[0]    
    
    plt.rcParams['figure.figsize'] = (num_examples * size_mult, num_iterations * size_mult)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
    for c in range(num_iterations):
        for i, ex in enumerate(examples):
            plt.subplot(num_iterations, num_examples, num_examples * c + i + 1)            
            plt.imshow(ex[:,:,c])  
            plt.axis('off')
            if c == 0 and i == 0 and title is not None:
                # only way I found to keep title placement 
                # semi-consistent for different channel counts
                plt.text(
                    x=0,
                    y=-ex.shape[1] // 4 // size_mult,
                    s=title,
                    fontsize=13,
                    horizontalalignment='left', 
                    verticalalignment='bottom')

    plt.show()
    
    
def plot_augmented(examples, alpha_range=0, sigma=0, 
                   width_shift_range=0, height_shift_range=0, zoom_range=0.0, 
                   iterations=1, title=None, size_mult=1):
    """Plot output after elastic distortion and select Keras data augmentations.
    
    # Arguments
        examples: Numpy array with shape (num_examples, height, width, num_iterations).
        alpha_range, sigma: arguments for `elastic_transform()`.
        width_shift_range, height_shift_range, zoom_range: arguments for Keras `ImageDataGenerator()`.
        iterations: Int, number of times to randomly augment the examples.
        title: Plot title string.
        size_mult: Multiply figsize by `size_mult`.
    """
    
    datagen = keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=width_shift_range, 
        height_shift_range=height_shift_range, 
        zoom_range=zoom_range,  
        preprocessing_function=lambda x: elastic_transform(x, alpha_range=alpha_range, sigma=sigma)
    )
    x = [datagen.flow(examples, shuffle=False).next() for i in range(iterations)]
    x = np.concatenate(x, axis=-1)
    plot_digits(x, title=title, size_mult=size_mult)
num_examples = 10
ed_examples = np.expand_dims(x_train[np.random.choice(x_train.shape[0], num_examples)], -1)

plot_digits(ed_examples, title='Input Images')

# elastic distortion
plot_augmented(ed_examples, alpha_range=8, sigma=2, 
               iterations=3, title='Elastic Distortion | alpha=8, sigma=2')
plot_augmented(ed_examples, alpha_range=8, sigma=3, 
               iterations=3, title='Elastic Distortion | alpha=8, sigma=3')
plot_augmented(ed_examples, alpha_range=10, sigma=3,
               iterations=3, title='Elastic Distortion | alpha=10, sigma=3')
num_examples = 6
b_examples = np.expand_dims(x_train[np.random.choice(x_train.shape[0], num_examples)], -1)

plot_digits(b_examples, title='Input Images', size_mult=2)

# shift
plot_augmented(b_examples, width_shift_range=2, height_shift_range=2,
               title='Integer Shift', size_mult=2)
plot_augmented(b_examples, width_shift_range=1., height_shift_range=1.,
               title='Float Shift', size_mult=2)

# elastic distortion & shift
plot_augmented(b_examples, alpha_range=[8, 10], sigma=3, 
               width_shift_range=2, height_shift_range=2,
               title='Elastic Distortion and Integer Shift', size_mult=2)
plot_augmented(b_examples, alpha_range=[8, 10], sigma=3, 
               width_shift_range=1., height_shift_range=1.,
               title='Elastic Distortion and Float Shift', size_mult=2)
num_examples = 10
examples = np.expand_dims(x_train[np.random.choice(x_train.shape[0], num_examples)], -1)

plot_digits(examples, title='Input Images')

plot_augmented(examples, alpha_range=[8, 10], sigma=3, 
               width_shift_range=2, height_shift_range=2, zoom_range=0, 
               iterations=10, title='Elastic Distortion and Integer Shift | alpha_range=[8, 10], sigma=3, shift_range=2')