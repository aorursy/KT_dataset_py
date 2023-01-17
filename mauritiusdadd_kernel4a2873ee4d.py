import multiprocessing

from functools import partial 

import numpy as np

import matplotlib.pyplot as plt

import scipy.optimize

%matplotlib inline



SUN_RADIUS = 1

JUPITER_RADIUS = (1/9.735)*SUN_RADIUS



def star_image(a1=0, a2=0, size=4096):

    x_lin = np.linspace(-1, 1, size)

    y_lin = np.linspace(-1, 1, size)

    X, Y = np.meshgrid(x_lin, y_lin)



    sin_theta = ((X**2 + Y**2)**2)

    

    valid = abs(sin_theta) <= 1

    angle = np.arcsin(sin_theta, where=valid)

    return valid*(1 + a1*(np.cos(angle) - 1) + a2*(np.cos(angle)**2 - 1))



def planet_shadow(x, b, planet_radius, size=4096):

    x_lin = np.linspace(-1, 1, size)

    y_lin = np.linspace(-1, 1, size)

    X, Y = np.meshgrid(x_lin, y_lin)

    r = ((X - x)**2 + (Y - b)**2)**0.5

    return r > planet_radius



def transit(x_data, jd0, duration, Rp,b, a1=0, a2=0, Rs=0.81*SUN_RADIUS, size=512):

    

    bkgnd = star_image(a1, a2, size=size)

    total_flux = bkgnd.sum()



    Rp /= Rs

    

    duration /= 60*24

    x_points = (x_data - jd0)/(duration)

    n_points = len(x_points)



    flux = np.zeros((2, n_points))

    

    shadow = planet_shadow(0, b, Rp, size=size)

    shadow = np.pad(shadow, ((0, 0), (size, size)), constant_values=1)

    flux = np.zeros_like(x_data)

    for i, x in enumerate(x_points):

        i_shift = int(x*size*np.cos(b))

        shifted = np.roll(shadow, i_shift)

        shadow_casted = bkgnd*shifted[:, size:-size]

        flux[i] = 2.5*np.log10((shadow_casted).sum()/total_flux)

    return flux



def fit(x_data, y_data, star_radius, resolution=512):

    # computing initial guesses

    baseline = y_data[:5].mean()

    

    delta = (baseline + y_data.min())/2

    jd0 = data[0][data[1]<=delta]

    duration = (jd0.max() - jd0.min())*(60*24)

    jd0 = (jd0.max()+jd0.min())/2

    R0 = (10**(baseline/2.5) - 10**(y_data.min()/2.5))**0.5

    R0 /= JUPITER_RADIUS

    

    print(

        "--- Stime iniziali ---\n"

        f"Durata del transito: {duration:.2f} minuti\n",

        f"Raggio del pianeta: {R0:.3f} R_giove\n",

        f"Inclinazione orbita: 0°\n",

        f"Parametro d'impatto: 0.0\n",

        f"Parametri limb-darkening: a1=0, a2=0\n",

    )

    

    def loss(params, yy, xx):

        return y_data - transit(xx, *params, Rs=star_radius, size=resolution)

    

    p0 = (jd0, 1.1*duration, 0.5*JUPITER_RADIUS, 0, 0.0, 0.0)

    res = scipy.optimize.least_squares(loss, p0, args=(y_data, x_data), ftol=1e-13, xtol=1e-13, diff_step=0.001, max_nfev=5000)



    return res['x']

# INPUT_FILE = '../input/wasp_10_b_2.csv'

INPUT_FILE = '../input/wasp_10_b.csv'



data = np.genfromtxt(INPUT_FILE, delimiter=',')[1:]

data = data[data[..., 1] < 99].T

data[1] -= data[1][:5].mean()

data[1] = -data[1]



parameters = fit(data[0], data[1], 0.81*SUN_RADIUS )



plt.plot(data[0], data[1], 'o', data[0], transit(data[0], *parameters), '-', linewidth=3)



print(

    "--- Risultati del fit ---\n"

    f"Durata del transito: {parameters[1]:.2f} minuti\n",

    f"Raggio del pianeta: {parameters[2]/JUPITER_RADIUS:.3f} R_giove\n",

    f"Inclinazione orbita: {np.arccos(abs(parameters[3]))*180/np.pi:.1f} °\n",

    f"Parametro d'impatto: {abs(parameters[3]):.3f}\n",

    f"Parametri limb-darkening: a1={parameters[4]:.4f}, a2={parameters[5]:.4f}\n",

)