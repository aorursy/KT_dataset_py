# Importing required libraries

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

%matplotlib inline
def IF_ADAMS(x, y, vals):

    '''

    Enter the x, y and vals values to generate a curve that may be imparted on a MOTION or FORCE.

    x: x coodinates

    y: y coordinates

    vals: The actual values of motion in mm or degrees, or radians

    

    SAMPLE INPUT 1: Rotation

    u, v: x, y coordinates fo curve

    u = [i for i in np.arange(0, 3.5, 0.5)]

    v = [0, 1, 0.764, 0.764, 0.764, 0, 0.764]

    val = '85d'

    IF_ADAMS(a, b, val)

    

    SAMPLE INPUT 2: Translation

    u = [0.0, 1.0, 1.5, 2.0, 3.0]

    v = [0.0, 0.0, 1.0, 0.0, 0.0]

    val = '31.114'

    IF_ADAMS(u, v, val)

    

    NOTE:

    Please enter the maximum value for the input 'vals', i.e. if you need to move something by 100mm or 100 degrees, then give, 100 or 100d as the values for 'val'

    Adjust the values of the y to get into different output values of the movement

    '''

    for i in range(len(x) - 1):

        x_new = [x[i], x[i + 1]]

        y_new = [y[i], y[i + 1]]



        l = np.polyfit(x_new, y_new, 1)



        func = np.poly1d(l, variable = 'time')

        if len(list(func.coef)) < 2:        

            func_text = str(round(list(func.coef)[0], 3))

        else:

            func_text = str(round(list(func.coef)[0], 3)) + ' * time + ' + str(round(list(func.coef)[1], 3))

        

        print(func)

        print(list(zip(x_new, y_new)))



        plt.scatter(x_new, y_new)

        plt.plot(x_new, y_new)

            

        if i == 0:

            to_adams = 'IF(time - ' + str(x[i + 1]) + ':' + ' <VALUE>*(' + func_text + '),' + ' <VALUE>*(' + func_text + '),'

        elif i > 0 & i < (len(x) - 1):

            to_adams = to_adams + ' IF(time - ' + str(x[i + 1]) + ':' + ' <VALUE>*(' + func_text + '),' + ' <VALUE>*(' + func_text + '),'

        elif i == (len(x) - 1):

            to_adams = to_adams + ' <VALUE>*(' + func_text + ')'

    

    to_adams = to_adams + ' <VALUE>*(' + func_text + ')'

    to_adams = to_adams + (')' * (len(x) - 1))

    to_adams = to_adams.replace('<VALUE>', vals)

    

    return to_adams
a = [i for i in np.arange(0, 3.5, 0.5)]

b = [0, 1, 0.764, 0.764, 0.764, 0, 0.764]

val = '85d'



text = IF_ADAMS(a, b, val)

text
plate_a = [0.0, 1.0, 1.5, 2.0, 3.0]

plate_b = [0.0, 0.0, 1.0, 0.0, 0.0]

plate_val = '31.114'



IF_ADAMS(plate_a, plate_b, plate_val)