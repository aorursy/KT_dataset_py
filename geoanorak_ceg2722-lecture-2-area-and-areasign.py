# This workbook leads directly to assessment number 1 in this module
# Copy or type out the commands in your own notebook
import shapely.geometry as shg
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 12, 12 #makes our plots bigger!
#Create some polygons
p1 = [(1.0, 1.0), (1.0, 5.0), (5.0, 5.0), (5.0, 1.0), (1.0, 1.0)]
p2 = [(2, 1), (5, 1), (6, 3), (6, 4),(3, 8),(1, 4), (1, 2), (2, 1)]

#Visualise them
xy1 = list(zip(*p1))
xy2 = list(zip(*p2))

plt.plot(xy1[0], xy1[1])
plt.plot(xy2[0], xy2[1])

#Check areas using shapely
sp1 = shg.Polygon(p1)
sp2 = shg.Polygon(p2)
print('Area of polygon 1 {} is {}'.format(sp1, sp1.area))
print('Area of polygon 2 {} is {}'.format(sp2, sp2.area))
#Calculate the area of a polygon using vector product
def newarea(poly):
    sumarea = 0
    for i in range(0, len(poly)-1):
        xi = poly[i][0]
        yi = poly[i][1]
        xj = poly[i+1][0]
        yj = poly[i+1][1]
        sumarea = sumarea + ((xi*yj)-(yi*xj))
    sumarea = (sumarea) * 0.5
    return sumarea
print('Area of polygon 1 is {}'.format(newarea(p1)))

#Switch order of polygon vertices using the REVERSE function and we now get +16
print ('Polygon 1: {}'.format(p1))
p1.reverse()
print ('Polygon 1: {}'.format(p1))
print('Area of polygon 1 is {}'.format(newarea(p1)))
print('Area of polygon 2 is: {}'.format(newarea(p2)))
#So that works now lets change the function so that it provides just the sign rather than the actual area

def areasign(poly):
    sumarea = 0
    for i in range(0, len(poly)-1):
        xi = poly[i][0]
        yi = poly[i][1]
        xj = poly[i+1][0]
        yj = poly[i+1][1]
        sumarea = sumarea + ((xi*yj)-(yi*xj))
        
    if sumarea == 0: return 0
    if sumarea < 0:  return -1
    if sumarea > 0:  return 1
#And test it
print('Areasign of polygon 1 is: {}'.format(areasign(p1)))
p1.reverse()
print('Areasign of polygon 1 is: {}'.format(areasign(p1)))
#test the handedness of a point (does it lie to the left, right or collinear with a line segment)
#see slide 11 in lecture 2 p, p' and p'' and line qr

testline = [(2,2), (5,5)]
tp1 = (1,5)
tp2 = (6,2)
tp3 = (1,1)
#Plot them out so we can visualise them
#tp1 is to the left of the line (the line is going up)
#tp2 is to the right of the line
#tp3 is collinear with the line

tl1xy = list(zip(*testline))
plt.plot(tl1xy[0], tl1xy[1])
plt.plot(tp1[0], tp1[1], 'rx')
plt.plot(tp2[0], tp2[1], 'bx')
plt.plot(tp3[0], tp3[1], 'gx')
# Need to create triangles representing
# pqr p'qr and p''qr

#Create a copy of the line
tri1 = testline.copy()
tri2 = testline.copy()
tri3 = testline.copy()

#insert the test point at the start
tri1.insert(0, tp1)
tri2.insert(0, tp2)
tri3.insert(0, tp3)

#Close the polygon by adding at the start coordinate at the end
tri1.append(tp1)
tri2.append(tp2)
tri3.append(tp3)

print(tri1)
print(tri2)
print(tri3)

print(areasign(tri1)) 
print(areasign(tri2))
print(areasign(tri3))