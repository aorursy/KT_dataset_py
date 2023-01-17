# Copy/type this code into your own notebook
# To get back to speed with some basic python skills
# To practice using the Kaggle notebook
# import the shapely library and drawing library
import shapely.geometry as shg
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 15, 15 #makes our plots bigger!
apoint = (1, 5) #point stored as coordinate pair in a tuple

print (apoint[0]) #the x value
print (apoint[1]) #the y value
print (apoint) #print the tuple
#A line is list of tuples
aline = [(1,1), (7, 2), (9, 6), (11, 8)]
#to view the line we need lists of x and y values so we use the ZIP command
xy = list(zip (*aline))
print (xy)
plt.plot(xy[0], xy[1])
print(aline)
print(aline[0]) #the 1st coordinate
print(aline[1], aline[2]) #line segment between 2 and 3rd vertices
print(aline[3][0]) #x value of 4th point
print(aline[3][1]) #y value of 4th point
#A Polygon looks just the same as a line
p = [(1.0, 1.0), (1.0, 5.0), (5.0, 5.0), (5.0, 1.0), (1.0, 1.0)]
print(p)
print(p[1])
print(p[3][1])

xy = list(zip(*p))
plt.plot(xy[0], xy[1])
#The shapely library
p1 = shg.Point(apoint) #Create a point note CAPS on function
p2 = shg.Point((5,9))  #Create another point
print(p1.distance(p2)) #Calculate distance between them
line1 = shg.LineString(aline) #lines are called linestrings in shapely (and elsewhere!)
print(line1.length)
line2 = shg.LineString([(5, 5), (7, 1), (8, 0)])
print(line1.intersects(line2))  # Test do they intersect
print(line1.intersection(line2)) # Point of intersection
#Check visually using the coords.xy function in Shapely to give us lists of x and y values
plt.plot(line1.coords.xy[0], line1.coords.xy[1])
plt.plot(line2.coords.xy[0], line2.coords.xy[1])
pt = line1.intersection(line2)                 #get the intersection point
plt.plot(pt.x, pt.y, 'ro')                     #plot it
#Polygons
pol1 = shg.Polygon(p)
pol2 = shg.Polygon([(4,4), (7,4), (7,8), (4,7), (4,4)])
print(pol1.area)
print(pol1.length)
print(pol1.intersects(pol2))
inter = pol1.intersection(pol2)
print(inter)
testpt1 = shg.Point(3,3)
testpt2 = shg.Point(7,7)
print(pol1.contains(testpt1)) #Point in Polygon test
print(pol1.contains(testpt2))
#And plotting them
#Polygons can have holes (interior rings) and the boundary is the exterior ring
plt.plot(pol1.exterior.xy[0], pol1.exterior.coords.xy[1])
plt.plot(pol2.exterior.coords.xy[0], pol2.exterior.coords.xy[1])
plt.fill(inter.exterior.coords.xy[0], inter.exterior.coords.xy[1], color='lightblue')
A = (1,1)
B = (5,1)
C = (2,5)

a = shg.Polygon([A, B, C])
print ("The area of the triangle is {} (Shapely calculation)".format(a.area))

#Carry out the calculation manually
area1 = A[0]*B[1]-A[1]*B[0]
area2 = B[0]*C[1]-B[1]*C[0]
area3 = C[0]*A[1]-C[1]*A[0]
area = 0.5 * (area1+area2+area3)
print ("the area of the triangle {}, {}, {} is {}".format(A, B, C, area))
