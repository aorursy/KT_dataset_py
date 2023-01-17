
#Filename for event
fn="train_1/event000001000"

#Row in particle file
n=100

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

A=np.loadtxt("../input/"+fn+"-truth.csv",skiprows=1,delimiter=',')
B=np.loadtxt("../input/"+fn+"-particles.csv",skiprows=1,delimiter=',')


#particle id to nth row in particle file.
part_id = B[n,0]

#Find hits from particle.
hits_from_part = np.argwhere(A[:,1]==part_id)[:,0]

#Print particle id and number of hits found.
print("PARTICLE ID:")
print(int(part_id))
print("Num hits: " + str(len(hits_from_part)))
print("")
#Get coordinates from hit-data.
coords = np.zeros((len(hits_from_part),3))
for i in range(0,len(hits_from_part)):
	coords[i,:] = A[hits_from_part[i],2:5]
	print("hit id: "+str(int(A[hits_from_part[i],0]))+ ". Absolute momentum: " + str(np.sqrt(np.sum(A[hits_from_part[i],5:8]**2))))
	
#Sort coordinates by z-component.	
idx = np.argsort(coords[:,2])
coords2=coords[idx]

#Plot
ax.plot(coords2[:,0],coords2[:,1],coords2[:,2],'.-')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Particle ID: " +str(int(part_id)))
plt.show()

