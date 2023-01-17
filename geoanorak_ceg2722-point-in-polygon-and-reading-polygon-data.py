import shapely.geometry as shg
import matplotlib.pyplot as plt
from pylab import rcParams

rcParams['figure.figsize'] = 8, 8 #makes our plots bigger!

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

def create_triangle(test_pt, line_seg):
    """ Creates a closed triangle from a test point and a line segment
    
    Parameters
    ----------
    test_pt : the test point in the format (x, y)
    line_seg: the line segment in the format [(x, y), (x, y)]
    
    Returns
    -------
    A triangle consisting of 4 coordinate tuples
    """
    tri = line_seg.copy()
    
    tri.insert(0, test_pt)
    tri.append(test_pt)
    
    return tri

def line_intersection(line1, line2):
    """ Tests whether 2 line segments intersect
    
    Parameters
    ----------
    line1 : the first line segment in the format [(x, y), (x, y)]
    line2 : the second line segment in the format [(x, y), (x, y)] 
    
    Returns
    -------
    True if the segmenets intersect,otherwise False
    """
    
    #create 4 triangles from the
    #start and end points of each line and the other line
    tri1 = create_triangle(line2[0], line1)
    tri2 = create_triangle(line2[1], line1)
    tri3 = create_triangle(line1[0], line2)
    tri4 = create_triangle(line1[1], line2)
    
    #Calculate the signs of the  areas of these triangles
    tri1sign = areasign(tri1)
    tri2sign = areasign(tri2)
    tri3sign = areasign(tri3)
    tri4sign = areasign(tri4)
    
    #if the signs are not equal then the lines intersect
    if ((tri1sign != tri2sign) and (tri3sign != tri4sign)):
        return True
    else:
        return False
# A polygon
p = [(1.0, 1.0), (1.0, 5.0), (5.0, 5.0), (5.0, 1.0), (1.0, 1.0)]

#the test point
test_pt = (3, 3)


# This point lies INSIDE the polygon
xy = list(zip(*p))
plt.plot(xy[0], xy[1])
plt.plot(test_pt[0], test_pt[1], 'ro')

#A halfline is a line segment from our test point to infinity
#infinity is hard to we just pick a big number

half_line = [test_pt, (1000000, 1000001)] 

#Why 1000001?


# The half line (truncated here!)
xy = list(zip(*p))
plt.plot(xy[0], xy[1])
plt.plot(test_pt[0], test_pt[1], 'ro')
plt.plot([3, 25], [3, 26], 'g-')
plt.text (15, 15, s="half line", rotation = 45, fontsize = 15)
# To iterate over a polygon we use a loop
# You might recognise this - it is the same code as as in the Area and Areasign functions
print(p)
counter = 1
for i in range(len(p)-1):
    print ('Line segment {} is ({}, {})'.format(counter, p[i], p[i+1]))
    counter = counter + 1
10 % 2
10 % 3
11 % 2
100 % 2
# Skeleton code for Point in Polygon function

"""
def pointinpolygon(pt, poly):
    #Create a half-line from the test point to infinity (or big number)
    
    #set a counter to 0
    
    #loop over the polygon
        #for each line segment check line intersection
        #if it intersects add to the counter
    
    #check the modulus of counter
    #if zero return False, otherwise return True
"""    
from shapely import wkt
list_of_polygons = []   #to store the polygons we read in our format

f = open ('../input/vector/test_polygons1.txt') #Open the file (mine is stored in the subfolder vector)

#iterate over the file
for each_line in f: 
    poly = wkt.loads(each_line)           #read WKT format into a Polygon (it recognises the type)
    print(list(poly.exterior.coords))     #use coords to generate lists in our format
    list_of_polygons.append(list(poly.exterior.coords))         #add the polygon to our list of polygons

display(list_of_polygons)
# And we can do the same with points

list_of_points = []   #to store the polygons we read in our format

f = open ('../input/vector/test_points1.txt') 

#iterate over the file
for each_line in f: 
    pt = wkt.loads(each_line)           
    print(list(pt.coords))     
    list_of_points.append(pt.coords[0])  #So we get a list of coordinates
display(list_of_points)
#Let's see what we have

for poly in list_of_polygons:
    xy = list(zip(*poly))
    plt.fill(xy[0], xy[1], alpha=0.5)
    
for pt in list_of_points:
    #print(pt)
    plt.plot(pt[0], pt[1], 'bx', ms=12)
import fiona                             #a library that reads spatial formats
from shapely.geometry import shape

shapefile_polygons = []                  #a list to store the polygons we load


c = fiona.open('../input/shapefiles/test_polygon_shapefile.shp') #Open the shapefile

for each_poly in c:
    geom = shape(each_poly['geometry'])
    poly_data = each_poly["geometry"]["coordinates"][0]  #EXTERIOR RING ONLY
    poly = shg.Polygon(poly_data)
    print(poly)
    #Converts into the same format as the other data
    shapefile_polygons.append(list(poly.exterior.coords))

display(shapefile_polygons)

# And read the points from a shapefile
shapefile_points = []

c = fiona.open('../input/shapefiles/test_point_shapefile.shp')

for each_pt in c:
    geom = shape(each_pt['geometry'])
    pt_data = each_pt["geometry"]["coordinates"]
    pt = shg.Point(pt_data)
    print(pt)
    shapefile_points.append(pt.coords[0])

display(shapefile_points)
# and we can plot these out
plt.gca().set_aspect('equal', adjustable='box')  #To make the x and y axis the same scale

for poly in shapefile_polygons:
    xy = list(zip(*poly))
    plt.fill(xy[0], xy[1], alpha=0.5)
    
for pt in shapefile_points:
    plt.plot(pt[0], pt[1], 'bx', ms=12)
import folium
import geopandas as gpd

#read the files using geopandas
polys=gpd.read_file('../input/shapefiles/test_polygon_shapefile.shp')
points = gpd.read_file('../input/shapefiles/test_point_shapefile.shp')

#Find the centroid to position the map
centroidp = polys.geometry.centroid 
#Convert from CRS British national grid to WGS84
wgspolys = polys.to_crs("EPSG:4326").to_json()
wgspoints = points.to_crs("EPSG:4326").to_json()
wgscentre = centroidp.to_crs("EPSG:4326")

long = wgscentre.x
lat = wgscentre.y

#Find spatial mean
long = long.mean()
lat = lat.mean()

#Plot it
mymap = folium.Map(location=[lat, long], tiles='openstreetmap', zoom_start=13)
plotpoly =folium.features.GeoJson(wgspolys)
plotpts = folium.features.GeoJson(wgspoints)
mymap.add_child(plotpoly)
mymap.add_child(plotpts)