%matplotlib notebook
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Orion
x = [-0.41, 0.57, 0.07, 0.00, -0.29, -0.32,-0.50,-0.23, -0.23]
y = [4.12, 7.71, 2.36, 9.10, 13.35, 8.13, 7.19, 13.25,13.43]
z = [2.06, 0.84, 1.56, 2.07, 2.36, 1.72, 0.66, 1.25,1.38]

print('x range:',min(x),max(x),
      "\ny range:",min(y),max(y),
      "\nz range:",min(z),max(z))
fig = plt.figure()
fig.add_subplot(1,1,1)

plt.scatter(x,y,c="brown",label = "Pretty cool constellation")
plt.title("2d Orion, the Metallica song",loc="left")
plt.legend()

ax=plt.subplot()
ax.set_xticks([i/10 for i in range(-6,7)])
ax.set_yticks([i for i in range(2,16,2)])

plt.xlabel("X Coordinates")
plt.ylabel("Y Coordinates")

plt.savefig("metallica1989.png")
plt.show()
fig_3d = plt.figure()
projection="3d"
ax = fig_3d.add_subplot(1,1,1,projection = "3d")
constellation3d = ax.scatter(x,y,z,alpha=1,c="black",linewidths=3,edgecolors="black",label='AMAZING constellation!!')
plt.legend()


plt.title("Orion - Metallica, 2020 Live.. oh and 3D plot of the Orion constellation!!")
ax.set_xlabel("X Coordinates")
ax.set_ylabel("Y Coordinates")
ax.set_zlabel("Z Cooridnates")

plt.savefig("metallica2020.png")
plt.show()
