from ipywidgets import widgets, Layout, Box

import matplotlib.pyplot as plt

from mpl_toolkits import mplot3d

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.lines as mlines

import numpy as np
def parallelProjectPoint(p, dirVector):

    """Performs a parallel projection for a point p based on a 

    given directing vector.

    

    Args:

        param p (float array): The point as 3D numpy array.

        param dirVector (float array): The direction vector used for the projection.



    Returns:

        A 3D point which contains the coordinates for the projection

        of point p on the X - Y plane

    """

    a = np.array([

        [1, 0, dirVector[0]],

        [0, 1, dirVector[1]],

        [0, 0, dirVector[2]]

    ])

    

    b = np.array([ p[0], p[1], p[2] ])



    x = np.linalg.solve(a, b)



    return [x[0], x[1], 0]



def centralProjectPoint(p, camera):

    """Performs a central projection for a point p based on a 

    given camera (3D point).

    

    Args:

        param p (float array): The point as 3D numpy array.

        param camera (float array): The camera used for the projection.



    Returns:

        A 3D point which contains the coordinates for the projection

        of point p on the X - Y plane

    """

    

    a = np.array([

        [1, 0, -camera[0]],

        [0, 1, -camera[1]],

        [0, 0, -camera[2]]

    ])

    b = np.array([

        p[0] - camera[0],

        p[1] - camera[1],

        p[2] - camera[2]

    ])



    x = np.linalg.solve(a, b)



    return [(x[0]/x[2]), (x[1]/x[2]), 0]



def get3DLine(P, Q, color="black"):

    """Draws a line between two points P and Q in R3.

    

    Args:

        param P (float array): The start point of the line.

        param Q (float array): The end point of the line.

        param color (string): The color of the line. Default is black.



    Returns:

        None

    """

    return mplot3d.art3d.Line3D((P[0], Q[0]), (P[1], Q[1]), (P[2], Q[2]), color=color)



def drawLine(p, q, color="black"):

    """Draws a line between two points p and q in R2.

    Args:

        param p (float tuple): The starting point of the line.

        param q (float tuple): The ending point of the line.

        param color (string): The color of the line. Default is black.



    Returns:

        None

    """

    

    twoDplot = plt.gca()

    twoDplot.add_line(mlines.Line2D([p[0],q[0]], [p[1],q[1]], color=color, linestyle="-", zorder=1))



def getParallelEpiped():

    """Collects all points needed to draw a parallelepiped.

    All information is taken from the global variables (support vector + vector a, b and c).

    The points are calculated via vector addition.

    

    Args:

        None



    Returns:

        List of numpy arrays containing the information of all points (x,y and z) of the parallelepiped

    """

    a = sv

    b = sv + bv

    c = sv + bv + cv

    d = sv + cv

    e = sv + av

    f = sv + av + bv

    g = sv + av + bv + cv

    h = sv + av + cv

    

    return [a, b, c, d, e, f, g, h]



def isLinearDependant():

    """Helper function to determine if the global vectors a,b and c are linear independant.

    

    Args:

        None



    Returns:

        True if at least two of the provided vectors are linear dependant, false if not.

    """

    matrix = np.array([av, bv, cv])

    return np.linalg.det(matrix) == 0

    
# Define the layouts (horizontal and vertical)

columnLayout = Layout(display='flex', flex_flow='column', align_items='stretch', border='none', width='auto')

rowLayout = Layout(display='flex', flex_flow='row', align_items='stretch', border='none', width='auto')



# Define the input widgets for the supporting vector s

sv = widgets.HTMLMath(value="Definition of the support vector $s&#8407;$:")

sx = widgets.BoundedFloatText(value=4, min=-10.0, max=10.0, step=0.1, description='$x$:', disabled=False )

sy = widgets.BoundedFloatText(value=1, min=-10.0, max=10.0, step=0.1, description='$y$:', disabled=False )

sz = widgets.BoundedFloatText(value=4, min=-10.0, max=10.0, step=0.1, description='$z$:', disabled=False )



# Define the input widgets for the directing vector r

vr = widgets.HTMLMath(value="Definition of the directing vector $r&#8407;$:")

rx = widgets.BoundedFloatText(value=0, min=-10.0, max=10.0, step=0.1, description='$x$:', disabled=False )

ry = widgets.BoundedFloatText(value=-1, min=-10.0, max=10.0, step=0.1, description='$y$:', disabled=False )

rz = widgets.BoundedFloatText(value=3, min=-10.0, max=10.0, step=0.1, description='$z$:', disabled=False )



# Define the input widgets for the camera

vcam = widgets.HTMLMath(value="Definition of the camera $C$:")

camx = widgets.BoundedFloatText(value=6, min=-10.0, max=10.0, step=0.1, description='$x$:', disabled=False )

camy = widgets.BoundedFloatText(value=2, min=-10.0, max=10.0, step=0.1, description='$y$:', disabled=False )

camz = widgets.BoundedFloatText(value=15, min=-10.0, max=15.0, step=0.1, description='$z$:', disabled=False )



# Define the input widgets for the directing vector a

va = widgets.HTMLMath(value="Definition of vector $a&#8407;$:")

ax = widgets.BoundedFloatText(value=2, min=-10.0, max=10.0, step=0.1, description='$x$:', disabled=False )

ay = widgets.BoundedFloatText(value=0, min=-10.0, max=10.0, step=0.1, description='$y$:', disabled=False )

az = widgets.BoundedFloatText(value=0, min=-10.0, max=10.0, step=0.1, description='$z$:', disabled=False )



# Define the input widgets for the directing vector b

vb = widgets.HTMLMath(value="Definition of vector $b&#8407;$:")

bx = widgets.BoundedFloatText(value=2, min=-10.0, max=10.0, step=0.1, description='$x$:', disabled=False )

by = widgets.BoundedFloatText(value=2, min=-10.0, max=10.0, step=0.1, description='$y$:', disabled=False )

bz = widgets.BoundedFloatText(value=1, min=-10.0, max=10.0, step=0.1, description='$z$:', disabled=False )



# Define the input widgets for the directing vector c

vc = widgets.HTMLMath(value="Definition of vector $c&#8407;$:")

cx = widgets.BoundedFloatText(value=0, min=-10.0, max=10.0, step=0.1, description='$x$:', disabled=False )

cy = widgets.BoundedFloatText(value=1, min=-10.0, max=10.0, step=0.1, description='$y$:', disabled=False )

cz = widgets.BoundedFloatText(value=2, min=-10.0, max=10.0, step=0.1, description='$z$:', disabled=False )



# Define the input widgets for selecting the type of projection

typeDesc = widgets.HTML(value="Type of the projection:")

typeSelect = widgets.RadioButtons( options=['Parallel', 'Central'], disabled=False)



# Assign the controls to the given boxes by a given layout

boxS = Box(children=[sv, sx, sy, sz], layout=columnLayout)

boxR = Box(children=[vr, rx, ry, rz], layout=columnLayout)

boxC = Box(children=[vcam, camx, camy, camz], layout=columnLayout)



boxVa = Box(children=[va, ax, ay, az], layout=columnLayout)

boxVb = Box(children=[vb, bx, by, bz], layout=columnLayout)

boxVc = Box(children=[vc, cx, cy, cz], layout=columnLayout)



vectorBox = Box(children=[boxS, boxR, boxC], layout=rowLayout)

definitionBox = Box(children=[boxVa, boxVb, boxVc], layout=rowLayout)



box = Box(children=[vectorBox, definitionBox, typeDesc, typeSelect], layout=columnLayout)



# Display the controls

display(box)
# Define the plot figure

fig = plt.figure()

fig.set_size_inches(10,10)



threeDplot = fig.add_subplot(111, projection='3d', aspect='equal')

threeDplot.set_xlim(0,10)

threeDplot.set_ylim(0,10)

threeDplot.set_zlim(0,10)



threeDplot.set_xlabel('X')

threeDplot.set_ylabel('Y')

threeDplot.set_zlabel('Z')



# The origin, read the directing vector and the camera from the configuration. 

origin = np.array([0,0,0])

rv     = np.array([rx.value ,ry.value, rz.value])

cam    = np.array([camx.value, camy.value, camz.value])



# Definition of the support vector

sv     = np.array([sx.value, sy.value, sz.value])



# Definition of the vector named a

av     = np.array([ax.value, ay.value, az.value])



# Definition of the vector named b

bv     = np.array([bx.value, by.value, bz.value])



# Definition of the vector named c

cv     = np.array([cx.value, cy.value, cz.value])



if isLinearDependant():

    print("CAUTION: Your input is not a valid parallelepiped")



points = getParallelEpiped()



if typeSelect.value == 'Central':

    projection = centralProjectPoint

    projParam  = cam

    threeDplot.scatter(cam[0], cam[1], cam[2])

else:

    projection = parallelProjectPoint

    projParam  = rv

    threeDplot.add_line(get3DLine(origin, rv, color="black"))



projectedPoints = []



for i in points: 

    projectedPoints.append(projection(i, projParam))

    

threeDplot.add_line(get3DLine(origin, sv, color="purple"))



threeDplot.add_line(get3DLine(points[0], points[4], color="#04ADBF"))

threeDplot.add_line(get3DLine(projectedPoints[0], projectedPoints[4], color="#04ADBF"))

threeDplot.add_line(get3DLine(points[1], points[5], color="#04ADBF"))

threeDplot.add_line(get3DLine(projectedPoints[1], projectedPoints[5], color="#04ADBF"))

threeDplot.add_line(get3DLine(points[2], points[6], color="#04ADBF"))

threeDplot.add_line(get3DLine(projectedPoints[2], projectedPoints[6], color="#04ADBF"))

threeDplot.add_line(get3DLine(points[3], points[7], color="#04ADBF"))

threeDplot.add_line(get3DLine(projectedPoints[3], projectedPoints[7], color="#04ADBF"))



threeDplot.add_line(get3DLine(points[0], points[1], color="#F2B705"))

threeDplot.add_line(get3DLine(projectedPoints[0], projectedPoints[1], color="#F2B705"))

threeDplot.add_line(get3DLine(points[2], points[3], color="#F2B705"))

threeDplot.add_line(get3DLine(projectedPoints[2], projectedPoints[3], color="#F2B705"))

threeDplot.add_line(get3DLine(points[4], points[5], color="#F2B705"))

threeDplot.add_line(get3DLine(projectedPoints[4], projectedPoints[5], color="#F2B705"))

threeDplot.add_line(get3DLine(points[6], points[7], color="#F2B705"))

threeDplot.add_line(get3DLine(projectedPoints[6], projectedPoints[7], color="#F2B705"))



threeDplot.add_line(get3DLine(points[0], points[3], color="#F23207"))

threeDplot.add_line(get3DLine(projectedPoints[0], projectedPoints[3], color="#F23207"))

threeDplot.add_line(get3DLine(points[1], points[2], color="#F23207"))

threeDplot.add_line(get3DLine(projectedPoints[1], projectedPoints[2], color="#F23207"))

threeDplot.add_line(get3DLine(points[4], points[7], color="#F23207"))

threeDplot.add_line(get3DLine(projectedPoints[4], projectedPoints[7], color="#F23207"))

threeDplot.add_line(get3DLine(points[5], points[6], color="#F23207"))

threeDplot.add_line(get3DLine(projectedPoints[5], projectedPoints[6], color="#F23207"))



plt.show()
if isLinearDependant():

    print("CAUTION: Your input is not a valid parallelepiped")

    

drawLine(projectedPoints[0], projectedPoints[4], color="#04ADBF")

drawLine(projectedPoints[1], projectedPoints[5], color="#04ADBF")

drawLine(projectedPoints[2], projectedPoints[6], color="#04ADBF")

drawLine(projectedPoints[3], projectedPoints[7], color="#04ADBF")



drawLine(projectedPoints[0], projectedPoints[1], color="#F2B705")

drawLine(projectedPoints[2], projectedPoints[3], color="#F2B705")

drawLine(projectedPoints[4], projectedPoints[5], color="#F2B705")

drawLine(projectedPoints[6], projectedPoints[7], color="#F2B705")



drawLine(projectedPoints[0], projectedPoints[3], color="#F23207")

drawLine(projectedPoints[1], projectedPoints[2], color="#F23207")

drawLine(projectedPoints[4], projectedPoints[7], color="#F23207")

drawLine(projectedPoints[5], projectedPoints[6], color="#F23207")



plt.axis([0.0, 10.0, 0.0, 10.0])

plt.show()