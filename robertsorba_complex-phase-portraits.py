%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (9,6.75)
from matplotlib import colors
from ipywidgets import interact, fixed
# simple function to be plotted
def complexFunc(z):
    return (z + 1) * (z - 2) / (z + 3)
def showPhasePortrait(func=complexFunc, ax=None, boxbound=2, cmap='hsv', 
                      showModContour=False, f_mod=2, showPhaseContour=False, f_ph=4,  
                      alpha=0.1):
    """Create phase portrait of a complex function.
    
    Args
    ----
    func: The complex function to be plotted.
    ax (maplotlib axes object): Axes to show the picture on. Default of None will display on 
        the most recent set of axes.
    boxbound (positive float): Defines the limits of the complex plane. Real and Imaginary 
        axes will be plotted from +/- boxbound.
    cmap (string): Determine which color mapping to use for phase. Options are 'hsv' or 
        'twilight'. HSV will look like a rainbow and have red on the positive real number 
        line, blue on the negative real number line, yellow/green on the positive imaginary 
        line, and purple on the negative imaginary line. Twilight will have white on the 
        positive real number line, black on the negative real number line, blue on the 
        positive imaginary line, and red on the negative imaginary line.
    showModContour (bool): If True, logarithmic modulus contours will be shown in greyscale.
    f_mod (positive float): Frequency of modulus contours. Higher numbers will show more 
        contours.
    showPhaseContour (bool): If True, phase contours will be shown in greyscale.
    f_ph (positive float): Frequency of phase contours. Higher numbers will show more contours.
    alpha (float between 0 and 1): Transparency of greyscale contours. 
    """
    x = np.arange(-boxbound, boxbound, boxbound/400)
    X, Y = np.meshgrid(x, x)

    # calculate function
    Z = X + 1.j * Y
    surface = func(Z)

    # negative to induce 180 deg phase shift to make colormap red at zero phase
    phase = np.angle(-surface) 
    modulus = np.abs(surface)
    
    if ax is None:
        ax = plt.gca()
    extent = [-boxbound, boxbound, -boxbound, boxbound]
    ax.imshow(phase, cmap=cmap, origin='lowerleft', extent=extent)
    
    if showModContour:
        modContour = np.ceil(np.log(modulus) * f_mod) - np.log(modulus) * f_mod
        ax.imshow(modContour, cmap='Greys', origin='lowerleft', 
                   extent=extent, interpolation='bilinear', alpha=alpha)
        
    if showPhaseContour:
        phaseContour = np.ceil(phase * f_ph/np.pi) - phase * f_ph/np.pi
        ax.imshow(phaseContour, cmap='Greys', origin='lowerleft', 
                   extent=extent, interpolation='bilinear', alpha=alpha)
        

plt.figure()
i = interact(showPhasePortrait, boxbound=(1, 10, 1), f_mod=(1, 10, 1), f_ph=(1, 10, 1),
         cmap=["hsv", "twilight"], alpha=(0.05, 0.2, 0.02), 
         func=fixed(complexFunc), ax=fixed(None))
