from ipywidgets import widgets, Layout, Box, Output

import math

import matplotlib.pyplot as plt

import matplotlib.lines as mlines

import numpy as np

def drawLine(p, q, linestyle="-"):

    """Draws a line between two points p and q

    Args:

        param p (float tuple): The starting point of the line.

        param q (float tuple): The ending point of the line.

        param linestyle (str): The linestyle to use. Default is "-".



    Returns:

        None

    """

    

    ax = plt.gca()

    ax.add_line(mlines.Line2D([p[0],q[0]], [p[1],q[1]], color="lightgray", linestyle=linestyle, linewidth=1))

    

def drawSquare(matrix):

    """Draws a triangle between the points a, b and c. 

    Args:

        param a (float tuple): Point A of the triangle.

        param b (float tuple): Point B of the triangle.

        param c (float tuple): Point C of the triangle.



    Returns:

        None

    """

    plt.gca().add_patch(plt.Polygon(matrix, color="gray"))
def rotatePoint(p):

    """Rotates a point (x, y) by the value read from the configuration UI. 

    Args:

        param p (float tuple): Point which should be rotated.



    Returns:

        The rotated point as float tuple.

    """

    theta = np.radians(t1Slider.value)

    matrix = np.array(( (np.cos(theta), np.sin(theta)), (-np.sin(theta), np.cos(theta)) ))

    return matrix.dot(p)



def reflectOnX(p):

    """Reflects a point (x, y) along the X-Axis. 

    Args:

        param p (float tuple): Point which should be reflected.



    Returns:

        The reflected point as float tuple.

    """

    matrix = np.array(( (1, 0), (0, -1) ))

    return matrix.dot(p)



def reflectOnY(p):

    """Reflects a point (x, y) along the Y-Axis. 

    Args:

        param p (float tuple): Point which should be reflected.



    Returns:

        The reflected point as float tuple.

    """

    matrix = np.array(( (-1, 0), (0, 1) ))

    return matrix.dot(p)



def shearPointX(p):

    """Shears a point (x, y) along the X-Axis. 

    The amount is calculated based on the value configured in the UI.

    Args:

        param p (float tuple): Point which should be sheared.



    Returns:

        The sheared point as float tuple.

    """

    theta = np.radians(t3XSlider.value)

    matrix = np.array(( (1, np.tan(theta)), (0, 1) ))

    return matrix.dot(p)



def shearPointY(p):

    """Shears a point (x, y) along the Y-Axis. 

    The amount is calculated based on the value configured in the UI.

    Args:

        param p (float tuple): Point which should be sheared.



    Returns:

        The sheared point as float tuple.

    """

    theta = np.radians(t3YSlider.value)

    matrix = np.array(( (1, 0), (np.tan(theta), 1) ))

    return matrix.dot(p)



def stretchPointX(p):

    """Stretches a point (x, y) along the X-Axis. 

    The factor is read from the configuration made on the UI.

    Args:

        param p (float tuple): Point which should be stretched.



    Returns:

        The stretched point as float tuple.

    """

    matrix = np.array(( (t4XFactorSlider.value, 0), (0, 1) ))

    return matrix.dot(p)



def stretchPointY(p):

    """Stretches a point (x, y) along the Y-Axis. 

    The factor is read from the configuration made on the UI.

    Args:

        param p (float tuple): Point which should be stretched.



    Returns:

        The stretched point as float tuple.

    """

    matrix = np.array(( (1, 0), (0, t4YFactorSlider.value) ))

    return matrix.dot(p)



def transformPoint(p):

    """Applies all linear transformation available to a point (x, y). 

    Args:

        param p (float tuple): Point which should be transformed.



    Returns:

        The stretched point as float tuple.

    """

    

    if t1Enabled.value == True:

        p = rotatePoint(p)

        

    if t2ReflectX.value == True:

        p = reflectOnX(p)

        

    if t2ReflectY.value == True:

        p = reflectOnY(p)

        

    if t3XEnabled.value == True:

        p = shearPointX(p)

        

    if t3YEnabled.value == True:

        p = shearPointY(p)

        

    if t4XEnabled.value == True:

        p = stretchPointX(p)

    

    if t4YEnabled.value == True:

        p = stretchPointY(p)

    

    return p
tab_contents = ['T0', 'T1', 'T2', 'T3', 'T4']



# Configuration pannel for the range of the axis

t0Layout = Layout(display='flex', flex_flow='column', align_items='stretch', border='none', width='100%')

xRangeSlider = widgets.FloatRangeSlider(value=[-2, 4], min=-20, max=20.0, step=1, description='Abscissa:',disabled=False, continuous_update=False,orientation='horizontal',readout=True,readout_format='1',)

yRangeSlider = widgets.FloatRangeSlider(value=[-2, 4], min=-20, max=20.0, step=1, description='Ordinate:',disabled=False, continuous_update=False,orientation='horizontal',readout=True,readout_format='1',)

t0Box = Box(children=[xRangeSlider, yRangeSlider], layout=t0Layout)



# Configuration pannel for the rotation configuration

t1Layout = Layout(display='flex', flex_flow='column', align_items='stretch', border='none', width='100%')

t1Enabled = widgets.Checkbox(value=True, description='Clockwise rotation around the origin point:',disabled=False)

t1Slider = widgets.IntSlider(value=0, min=-360, max=360, step=1, description='Degree:', disabled=False, continuous_update=False, orientation='horizontal', readout=True, readout_format='d')

t1Box = Box(children=[t1Enabled, t1Slider], layout=t1Layout)



# Configuration pannel for the mirroring configuration

t2Layout = Layout(display='flex', flex_flow='column', align_items='stretch', border='none', width='100%')

t2ReflectX = widgets.Checkbox(value=False, description='Reflection along the abscissa',disabled=False)

t2ReflectY = widgets.Checkbox(value=False, description='Reflection along the ordinate',disabled=False)

t2Box = Box(children=[t2ReflectX, t2ReflectY], layout=t2Layout)



# Configuration pannel for the shearing configuration

t3Layout = Layout(display='flex', flex_flow='column', align_items='stretch', border='none', width='100%')

t3XEnabled = widgets.Checkbox(value=True, description='Shear parallel to abscissa:',disabled=False)

t3XSlider = widgets.IntSlider(value=0, min=-90, max=90, step=1, description='Degree:', disabled=False, continuous_update=False, orientation='horizontal', readout=True, readout_format='d')

t3YEnabled = widgets.Checkbox(value=True, description='Shear parallel to ordinate:',disabled=False)

t3YSlider = widgets.IntSlider(value=0, min=-90, max=90, step=1, description='Degree:', disabled=False, continuous_update=False, orientation='horizontal', readout=True, readout_format='d')

t3Box = Box(children=[t3XEnabled, t3XSlider, t3YEnabled, t3YSlider], layout=t3Layout)



# Configuration pannel for the stretch configuration

t4Layout = Layout(display='flex', flex_flow='column', align_items='stretch', border='none', width='100%')

t4XEnabled = widgets.Checkbox(value=True, description='Stretch parallel to abscissa:',disabled=False)

t4XFactorSlider = widgets.FloatSlider(value=1,min=0.01,max=5,step=0.01, description='Factor:',disabled=False,continuous_update=False,orientation='horizontal',readout=True,readout_format='.2f',)

t4YEnabled = widgets.Checkbox(value=True, description='Stretch parallel to ordinate:',disabled=False)

t4YFactorSlider = widgets.FloatSlider(value=1,min=0.01,max=5,step=0.01, description='Factor:',disabled=False,continuous_update=False,orientation='horizontal',readout=True,readout_format='.2f',)

t4Box = Box(children=[t4XEnabled, t4XFactorSlider, t4YEnabled, t4YFactorSlider], layout=t4Layout)



tab = widgets.Tab()

tab.children = [t0Box, t1Box, t2Box, t3Box, t4Box]





tab.set_title(0, "Axis")

tab.set_title(1, "Rotate")

tab.set_title(2, "Reflection")

tab.set_title(3, "Shear")

tab.set_title(4, "Stretch")



display(tab)
# Get the value range for the grid

# and add +/- 15. This ensures that the grid edges are not visible when

# applying rotation. 

# More complex transformations like shearing by almost 90 degrees makes the

# grid collapse.

xMin = math.floor(xRangeSlider.value[0]) - 15

xMax = math.ceil(xRangeSlider.value[1]) + 15

yMin = math.floor(yRangeSlider.value[0]) - 15

yMax = math.ceil(yRangeSlider.value[1]) + 15



# Draw the lines along the X-Axis

for i in range(xMin, xMax):

    drawLine(transformPoint((i, yMin)), transformPoint((i, yMax)))



# Draw the lines along the Y-Axis

for i in range(yMin, yMax):

    drawLine(transformPoint((xMin, i)), transformPoint((xMax, i)))



# Draw the square

drawSquare([transformPoint((0,0)),transformPoint((1,0)), transformPoint((1,1)), transformPoint((0,1))])

    

# And show the plot with the correct value range.

plt.axis([xRangeSlider.value[0], xRangeSlider.value[1], yRangeSlider.value[0], yRangeSlider.value[1]])

plt.show()