#import libraries
import numpy as np
import matplotlib.pyplot as plt
  
#area of the bounding box, square of side 2 meters.
box_area = 4.0
#number of samples or stones to trhow.
N_total = 1000000
#drawing random points uniform between -1 and 1
X = np.random.uniform(low=-1, high=1, size=N_total)  
Y = np.random.uniform(low=-1, high=1, size=N_total)   
# calculate the distance of the point from the center 
distance = np.sqrt(X**2+Y**2);  
# check if point is inside the circle (diameter 2 meters)   
is_point_inside = distance<1.0
# sum up the stones inside the circle
N_inside=np.sum(is_point_inside)
# estimate the circle area
circle_area = box_area * N_inside/N_total
# some nice visualization
plt.scatter(X[0:1000],Y[0:1000],  s=40,c=is_point_inside[0:1000],alpha=.6, edgecolors='None')  
plt.axis('equal')

# text output
print ('Area of the circle = ', circle_area)
print ('pi = ', np.pi)
plt.show()

