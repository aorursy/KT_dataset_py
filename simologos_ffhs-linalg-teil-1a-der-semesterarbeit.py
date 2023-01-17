from ipywidgets import widgets, Layout, Box

import matplotlib.pyplot as plt

import matplotlib.lines as mlines

import matplotlib.path as mplPath

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

    ax.add_line(mlines.Line2D([p[0],q[0]], [p[1],q[1]], color="black", linestyle=linestyle, zorder=1))

    

def drawTriangle(a, b, c):

    """Draws a triangle between the points a, b and c. 

    Args:

        param a (float tuple): Point A of the triangle.

        param b (float tuple): Point B of the triangle.

        param c (float tuple): Point C of the triangle.



    Returns:

        None

    """

    plt.gca().add_patch(plt.Polygon([a, b, c], color="gray", alpha=0.5, zorder=10))

    

def getRoundedPoint(p):

    """Rounds the coordinates of a point to two decimals

    Args:

        param p (float tuple): X and Y coordinates of the point to round its coordinates.



    Returns:

        A float tuple describing a point with rounded coordinates.

    """  

    return (round(p[0], 2), round(p[1], 2))



    

def isInTriangle(point, triangle):

    """Checks if a point lies inside a triangle. 

    Args:

        param point (float tuple): X and Y coordinates of the point used to run the test.

        param polygon (list of float tuple): The polygon as list of tuppels.



    Returns:

        True if the point is inside the polygon, False if not.

    """  

    

    # If the point is equal to an edge of the triangle, I count it as inside.

    for p in triangle:

        if getRoundedPoint(point) == getRoundedPoint(p):

            return True

    

    # Get all intersection points with the triangle as list.

    intersections = [

        getIntersection([point, (point[0] + 10, point[1])], [triangle[0], triangle[1]]),

        getIntersection([point, (point[0] + 10, point[1])], [triangle[0], triangle[2]]),

        getIntersection([point, (point[0] + 10, point[1])], [triangle[1], triangle[2]]),

    ]



    # Exclude all None values (getIntersection returns None if no intersection point found)

    nonNonePoints = [i for i in intersections if i]

    

    # Check if the point is exactly on a line defining the triangle.

    for p in nonNonePoints:

        if getRoundedPoint(point) == getRoundedPoint(p):

            return True

        

        

    # remove all intersections which are left from the point to test

    validIntersections = []

    for p in nonNonePoints:        

        if p[0] < point[0]:

            continue            

        validIntersections.append(p)        

            

    # If here, we are either completely "inside" or " outside the tringle"

    # therefore if the intersection points are odd, we are "inside"

    # the triangle.

    return len(validIntersections) % 2 == 1





def isBetween(p, q, r):

    """Checks if the point r lies inside the rectangel created by point p and q.

    This is used to test if point r is outside of the line between p and q, but does not

    ensure if r is on one line with p and q.

    

    Args:

        param point p (float tuple): X and Y coordinates of the point P.

        param point q (float tuple): X and Y coordinates of the point Q.

        param point r (float tuple): X and Y coordinates of the point R.



    Returns:

        True if the point r is in between point p and q.

    """

    minX = min(p[0], q[0])

    maxX = max(p[0], q[0])

    

    minY = min(p[1], q[1])

    maxY = max(p[1], q[1])

    

    if r[0] <= minX:

        return False

    

    if r[0] >= maxX:

        return False

    

    if r[1] <= minY:

        return False

    

    if r[1] >= maxY:

        return False

    

    return True

    

def det(a, b):

    """Calculates the determinant of a 2 times 2 matrix, submitted as two vectors a and b.

    

    Args:

        param vector a (float tuple): The first vector used to calculate the determinant.

        param vector b (float tuple): The second vector used to calculate the determinant.



    Returns:

        floating point number expressing the calculated determinant.

    """

    return a[0] * b[1] - a[1] * b[0]



def getIntersection(l1, l2):

    """Checks if two lines intersect with each other.

    

    Args:

        param vector l1 (list): List containing two tuples of floating point numbers which describe the first line.

        param vector l2 (list): List containing two tuples of floating point numbers which describe the second line.



    Returns:

        None if the lines do not intersect

        Tuple of floating point numbers describing the X and Y coordinates of the point

        where the two lines are intersecting.

    """

    dx = (l1[0][0] - l1[1][0], l2[0][0] - l2[1][0])

    dy = (l1[0][1] - l1[1][1], l2[0][1] - l2[1][1])



    div = det(dx, dy)



    if div == 0:

      return None



    d = (det(*l1), det(*l2))

    x = det(d, dx) / div

    y = det(d, dy) / div

    

    if isBetween(l2[0], l2[1], (x, y)) == False:

        return None

    

    return (x, y)
# Create the fields used to capture the triangles coordinates.

triangle_layout = Layout(display='flex', flex_flow='row', align_items='stretch', border='none', width='100%')



ax = widgets.FloatSlider(value=2.35, min=0, max=10.0, step=0.01, description="$Ax$:", disabled=False, continuous_update=False, orientation='vertical', readout=True, readout_format='.11',)

ay = widgets.FloatSlider(value=3.65, min=0, max=10.0, step=0.01, description="$Ay$:", disabled=False, continuous_update=False, orientation='vertical', readout=True, readout_format='.11',)

bx = widgets.FloatSlider(value=7.55, min=0, max=10.0, step=0.01, description="$Bx$:", disabled=False, continuous_update=False, orientation='vertical', readout=True, readout_format='.11',)

by = widgets.FloatSlider(value=4.25, min=0, max=10.0, step=0.01, description="$By$:", disabled=False, continuous_update=False, orientation='vertical', readout=True, readout_format='.11',)

cx = widgets.FloatSlider(value=4.56, min=0, max=10.0, step=0.01, description="$Cx$:", disabled=False, continuous_update=False, orientation='vertical', readout=True, readout_format='.11',)

cy = widgets.FloatSlider(value=7.65, min=0, max=10.0, step=0.01, description="$Cy$:", disabled=False, continuous_update=False, orientation='vertical', readout=True, readout_format='.11',)



triangle_box = Box(children=[ax, ay, bx, by, cx, cy], layout=triangle_layout)



display(triangle_box)

# Create the fields used to capture the lines coordinates.

line_layout = Layout(display='flex', flex_flow='row', align_items='stretch', border='none', width='100%')



px = widgets.FloatSlider(value=1.35, min=0, max=10.0, step=0.01, description="$Px$:", disabled=False, continuous_update=False, orientation='vertical', readout=True, readout_format='.11',)

py = widgets.FloatSlider(value=4.65, min=0, max=10.0, step=0.01, description="$Py$:", disabled=False, continuous_update=False, orientation='vertical', readout=True, readout_format='.11',)

qx = widgets.FloatSlider(value=7.55, min=0, max=10.0, step=0.01, description="$Qx$:", disabled=False, continuous_update=False, orientation='vertical', readout=True, readout_format='.11',)

qy = widgets.FloatSlider(value=6.25, min=0, max=10.0, step=0.01, description="$Qy$:", disabled=False, continuous_update=False, orientation='vertical', readout=True, readout_format='.11',)



line_box = Box(children=[px, py, qx, qy], layout=line_layout)



display(line_box)
# Collect the user input

a = (ax.value, ay.value)

b = (bx.value, by.value)

c = (cx.value, cy.value)

p = (px.value, py.value)

q = (qx.value, qy.value)



# Get the intersection points for all sides of the triangle

points = [p, getIntersection((a,b),(p,q)), getIntersection((a,c),(p,q)), getIntersection((b,c),(p,q)), q]



# If the line does not intersect with a specific side of the triangle, then None is returned.

# Ensure that None values are removed from the list.

nonNonePoints = [i for i in points if i]



# Now sort the points by their X-value followed by the Y-value

# in order to ensure that the full line is drawn correctly.

result = sorted(nonNonePoints , key=lambda k: [k[0], k[1]])



# Draw the line

count = len(result) - 1



for i in range(count):

    # Check if the start and end point of the line segment

    # lies inside the triangle.

    strInPoly = isInTriangle(result[i], [a, b, c,])

    endInPoly = isInTriangle(result[i+1], [a, b, c,])

    

    if strInPoly and endInPoly:

        # Yes, therefore draw a dotted line

        drawLine(result[i], result[i + 1], linestyle=":")

        continue

    

    # No, the line segment is outside the triangle,

    # draw a straight line.

    drawLine(result[i], result[i + 1])

    

# Draw the triangle

drawTriangle(a, b, c)



# Plot the axis and show the diagram.

plt.axis([0.0, 10.0, 0.0, 10.0])

plt.show()
