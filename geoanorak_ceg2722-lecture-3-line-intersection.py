# Intersection of 2 line segments
import shapely.geometry as shg
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 8, 8 #makes our plots bigger!
#Define the areasign code (note the doc comments!)    
def areasign(tri):
    """Finds the sign of the area of a closed triangle

    Parameters
    ----------
    tri : List
        List of coordinate tuples in the format
        [(x, y), (x, y). (x, y), (x, y)]
        First and last vertex are the same
    
    Returns
    -------
    int (-1, 1, 0)
        -1 if the triangle is encoded clockwise
        1  if the triangle is encoded anti-clockwise
        0  if the coordinates are a 1d line
    """
    sumarea = 0
    for i in range(0, len(tri)-1):
        xi = tri[i][0]
        yi = tri[i][1]
        xj = tri[i+1][0]
        yj = tri[i+1][1]
        sumarea = sumarea + ((xi*yj)-(yi*xj))
        
    if sumarea == 0: return 0
    if sumarea < 0:  return -1
    if sumarea > 0:  return 1
#Create a a line and copy 
testline = [(2,2), (5,5)]
testline = [(2,2), (5,5)]
tp1 = (1,5)
tp2 = (6,2)
tp3 = (1,1)

#PLOT THEM OUT
tl1xy = list(zip(*testline))
plt.plot(tl1xy[0], tl1xy[1])
plt.plot(tp1[0], tp1[1], 'rx')
plt.text(tp1[0] + 0.1, tp1[1], s ='pq', fontsize = 15)   #PLOTTING TEXT ONTO THE PLOT (x, y, "text")
plt.plot(tp2[0], tp2[1], 'bx')
plt.text(tp2[0] - 0.4, tp2[1], s ='p\'q', fontsize = 15) #Need to escape the ' character'
plt.plot(tp3[0], tp3[1], 'gx')
plt.text(tp3[0]+0.1 + 0.1, tp3[1]+0.1, s ='p\'\'q', fontsize = 15)
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
#FUNCTION TO CREATE TRIANGLE FROM LINE SEGMENT AND TEST POINT

def create_triangle(test_pt, line_seg):
    tri = line_seg.copy()
    
    tri.insert(0, test_pt)
    tri.append(test_pt)
    
    return tri
    
tri_test = create_triangle((1, 5), [(2,2), (5,5)])
print(tri_test)

xy = list(zip(*tri_test))
plt.plot(xy[0], xy[1], linewidth=4)                      #PLOT TRIANGLE
plt.fill(xy[0], xy[1], color = 'lightblue', alpha=0.5)   #FILL TRIANGLE
plt.plot(1, 5 , 'ro')                                    #PLOT THE TEST PT
plt.plot([2, 5], [2, 5], 'r-')                           #PLOT THE LINE SEGMENT
# 2 line segments

line1 = [(1,1), (5, 5)]
line2 = [(1, 4), (5, 1)]

xy1 = list(zip(*line1))
xy2 = list(zip(*line2))

plt.plot(xy1[0], xy1[1], color = 'red', linewidth = 4)
plt.plot(xy2[0], xy2[1], color = 'blue', linewidth =4)
#CREATE TRIANGLES FROM LINE1 and the start and endpoints of LINE2

tri1 = create_triangle(line2[0], line1)
tri2 = create_triangle(line2[1], line1)

print(areasign(tri1), areasign(tri2))
#PLOT THE TRIANGLES WE HAVE CREATED
xy1 = list(zip(*tri1))
plt.plot(xy1[0], xy1[1], linewidth=4, color='red')                      #PLOT TRIANGLE
plt.fill(xy1[0], xy1[1], color = 'lightblue', alpha=0.5)                #FILL TRIANGLE
xy2 = list(zip(*tri2))
plt.plot(xy2[0], xy2[1], linewidth=4, color='blue')                     #PLOT TRIANGLE
plt.fill(xy2[0], xy2[1], color = 'lightgreen', alpha=0.5)               #FILL TRIANGLE
# TO CHECK LINE SEGMENT INTERSECTION WE ALSO HAVE TO CHECK
# THE SIGNS OF THE TRIANGLES OF THE MADE FROM LINE2 AND THE 
# START AND END POINTS OF LINE 1

tri3 = create_triangle(line1[0], line2)
tri4 = create_triangle(line1[1], line2)

print(areasign(tri3), areasign(tri4))
#PLOT THE TRIANGLES WE HAVE CREATED
xy1 = list(zip(*tri1))
plt.plot(xy1[0], xy1[1], linewidth=4, color='red')                      #PLOT TRIANGLE
plt.fill(xy1[0], xy1[1], color = 'lightblue', alpha=0.5)                #FILL TRIANGLE
xy2 = list(zip(*tri2))
plt.plot(xy2[0], xy2[1], linewidth=4, color='blue')                     #PLOT TRIANGLE
plt.fill(xy2[0], xy2[1], color = 'lightgreen', alpha=0.5)               #FILL TRIANGLE
xy3 = list(zip(*tri3))
plt.plot(xy3[0], xy3[1], linewidth=4, color='coral')                    #PLOT TRIANGLE
plt.fill(xy3[0], xy3[1], color = 'coral', alpha=0.2)                    #FILL TRIANGLE
xy4 = list(zip(*tri4))
plt.plot(xy4[0], xy4[1], linewidth=4, color='magenta')                  #PLOT TRIANGLE
plt.fill(xy4[0], xy4[1], color = 'magenta', alpha=0.2)                  #FILL TRIANGLE

plt.text(1.55, 3.4, s = "Tri 1", fontsize=20, rotation=45)
plt.text(3.7, 1.7, s = "Tri 2", fontsize=20, rotation=225)
plt.text(2, 2, s = "Tri 3", fontsize=20, rotation=135)
plt.text(3.5, 3.5, s = "Tri 4", fontsize=20, rotation=-45)
#2 NEW LINES THAT DON'T INTERSECT
line3 = [(2,2), (5, 5)]
line4 = [(0.5, 2.5), (2.5, 0.5)]

xy1 = list(zip(*line3))
xy2 = list(zip(*line4))

plt.plot(xy1[0], xy1[1], linewidth=4, color='navy')
plt.plot(xy2[0], xy2[1], linewidth=4, color='orangered')
tri1 = create_triangle(line4[0], line3)
tri2 = create_triangle(line4[1], line3)

print(areasign(tri1), areasign(tri2))

tri3 = create_triangle(line3[0], line4)
tri4 = create_triangle(line3[1], line4)

print(areasign(tri3), areasign(tri4))
#PLOT THE TRIANGLES WE HAVE CREATED
xy1 = list(zip(*tri1))
plt.plot(xy1[0], xy1[1], linewidth=4, color='red')                      #PLOT TRIANGLE

xy2 = list(zip(*tri2))
plt.plot(xy2[0], xy2[1], linewidth=4, color='blue')                     #PLOT TRIANGLE

xy3 = list(zip(*tri3))
plt.plot(xy3[0], xy3[1], linewidth=4, color='coral', linestyle='dashed')                    #PLOT TRIANGLE

xy4 = list(zip(*tri4))
plt.plot(xy4[0], xy4[1], linewidth=4, color='magenta', alpha=0.8)                  #PLOT TRIANGLE



# ALL THAT'S LEFT IS TO WRAP THIS AS A FUNCTION
def line_intersection(line1, line2):
    #create 4 triangles from the
    #start and end points of each line and the other line
    tri1 = create_triangle(line2[0], line1)
    tri2 = create_triangle(line2[1], line1)
    tri3 = create_triangle(line1[0], line2)
    tri4 = create_triangle(line1[1], line2)
    
    #Find the sign of the area of each triangle
    tri1sign = areasign(tri1)
    tri2sign = areasign(tri2)
    tri3sign = areasign(tri3)
    tri4sign = areasign(tri4)
    #if the signs pf each pair are not the same then the lines intersect
    if ((tri1sign != tri2sign) and (tri3sign != tri4sign)):
        return True
    else:
        return False

print('These 2 cross')
print(line1)
print(line2)
print('These 2 do not cross')
print(line3)
print(line4)
print(line_intersection(line1, line2))
print(line_intersection(line3, line4))
