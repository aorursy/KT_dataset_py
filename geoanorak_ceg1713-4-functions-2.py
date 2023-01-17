import math 
r = 12.2  #radius
l = 9.71  #length (this is the letter "l" not the value 1!)
h = 6.591 #height
w = 7.25  #width

sa1 = math.pow(l/2, 2) + math.pow(h,2)
sa2 = w * math.sqrt(sa1)
sa3 = math.pow(w/2, 2) + math.pow(h, 2)
sa4 = l * math.sqrt(sa3)
sa_pyr = l * w + sa4 + sa2
print(sa_pyr)
# Function to calculate the surface area of a pyramid

def sa_pyramid(l, w, h):
    sa1 = math.pow(l/2, 2) + math.pow(h,2)
    sa2 = w * math.sqrt(sa1)
    sa3 = math.pow(w/2, 2) + math.pow(h, 2)
    sa4 = l * math.sqrt(sa3)
    sa_pyr = l * w + sa4 + sa2
    return sa_pyr # don't print it but pass the answer back

print(sa_pyramid(l, w, h))
print(sa_pyramid(l, 23, 13.7))

"""
Function to calculate the surface area of a pyramid
arguments - (l = length, w = width, h = height)
returns - the surface area
"""
def sa_pyramid(l, w, h):
    sa1 = math.pow(l/2, 2) + math.pow(h,2)
    sa2 = w * math.sqrt(sa1)
    sa3 = math.pow(w/2, 2) + math.pow(h, 2)
    sa4 = l * math.sqrt(sa3)
    sa_pyr = l * w + sa4 + sa2
    return sa_pyr # don't print it but pass the answer back
"""
Calculates distance between 2 planar coordinates
Using pythagora
Arguments - x and y of 1st coord (x1, y1) and x and y of 2nd coordinate(x2, y2)
Returns - the distance between coordinates
"""
def euclid(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    #a2 = b2 + c2 (pythagoras)
    d = math.pow(dx, 2) + math.pow(dy, 2)
    return math.sqrt(d)
    
print (euclid(1,1,5,5))
lat1 = 40.7128
lon1 = -74.0060
lat2 = 54.9783
lon2 = -1.6178
R = 6371
#Why the minus numbers - have a google if you don't know!

#We are using TRIG functions so MUST convert to Radians
dlat = math.radians(lat2 - lat1)
dlon = math.radians(lon2 - lon1)

rlat1 = math.radians(lat1)
rlat2 = math.radians(lat2)

a1 = math.pow(math.sin(dlat/2), 2)
a2 = math.cos(rlat1) * math.cos(rlat2)
a3 = math.pow(math.sin(dlon/2), 2)
a = a1 + a2 * a3

c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

d = R * c

print(d)

"""
Haversine formula to calculate Great Circle distance
between 2 lat/long coordinates
arguments (lat1, long1) = starting point
          (lat2, long2) = ending point
returns: Distance (in km)
"""
def great_circle_dist(lat1, long1, lat2, long2):
    R = 6371.0 # diameter of the earth in KM
    #We are using TRIG functions so MUST convert to Radians
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(long2 - long1)
    #print(dlat, dlon)
    rlat1 = math.radians(lat1)
    rlat2 = math.radians(lat2)

    a1 = math.pow(math.sin(dlat/2), 2)
    a2 = math.cos(rlat1) * math.cos(rlat2) 
    a3 = math.pow(math.sin(dlon/2),2)
    a = a1 + a2 * a3
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    d = R * c

    return d
print(great_circle_dist(lat1, lon1, lat2, lon2))
#Moscow
print(great_circle_dist(55.7558, 37.6173, lat2, lon2))
#Sydney
print (great_circle_dist(-33.8688, 151.2093, lat2, lon2))
#Bearings between 2 points 

# math.atan(dx/dy)

def bearing(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    b = math.atan(dx/dy)
    b = math.degrees(b)
    return b

bearing(1,1,5,5)
bearing(1,5,5,1)



