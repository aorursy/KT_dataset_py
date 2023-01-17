%matplotlib notebook

from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
# Orion

x = [-0.41, 0.57, 0.07, 0.00, -0.29, -0.32,-0.50,-0.23, -0.23]

y = [4.12, 7.71, 2.36, 9.10, 13.35, 8.13, 7.19, 13.25,13.43]

z = [2.06, 0.84, 1.56, 2.07, 2.36, 1.72, 0.66, 1.25,1.38]
fig = plt.figure(figsize=(6, 4))

fig.add_subplot(1,1,1)

plt.scatter(x,y,s=500,c=z,alpha=0.8,marker=(5,1),

            label="star")

plt.show()

# https://matplotlib.org/2.0.0/examples/pylab_examples/scatter_star_poly.html
fig_3d = plt.figure()

constellation3d = fig_3d.add_subplot(1,1,1,projection="3d")

constellation3d.scatter(x,y,z,s=500,c=z,alpha=0.8,marker=(5,1),

            label="star")

plt.show()

#Rotation works well in editing mode or on local computers but is lost in committed notebook.