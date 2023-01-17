[10.5, 5.2, 3.25, 7.0]
import numpy as np

video = np.array([10.5, 5.2, 3.25, 7.0])

video
video.size
video[2]  # 3rd element
%matplotlib inline

import matplotlib.pyplot as plt
u = np.array([2, 5])

v = np.array([3, 1])
x_coords, y_coords = zip(u, v)

plt.scatter(x_coords, y_coords, color=["r","b"])

plt.axis([0, 9, 0, 6])

plt.grid()

plt.show()
def plot_vector2d(vector2d, origin=[0, 0], **options):

    return plt.arrow(origin[0], origin[1], vector2d[0], vector2d[1],

              head_width=0.2, head_length=0.3, length_includes_head=True,

              **options)
plot_vector2d(u, color="r")

plot_vector2d(v, color="b")

plt.axis([0, 9, 0, 6])

plt.grid()

plt.show()
a = np.array([1, 2, 8])

b = np.array([5, 6, 3])
from mpl_toolkits.mplot3d import Axes3D



subplot3d = plt.subplot(111, projection='3d')

x_coords, y_coords, z_coords = zip(a,b)

subplot3d.scatter(x_coords, y_coords, z_coords)

subplot3d.set_zlim3d([0, 9])

plt.show()
def plot_vectors3d(ax, vectors3d, z0, **options):

    for v in vectors3d:

        x, y, z = v

        ax.plot([x,x], [y,y], [z0, z], color="gray", linestyle='dotted', marker=".")

    x_coords, y_coords, z_coords = zip(*vectors3d)

    ax.scatter(x_coords, y_coords, z_coords, **options)



subplot3d = plt.subplot(111, projection='3d')

subplot3d.set_zlim([0, 9])

plot_vectors3d(subplot3d, [a,b], 0, color=("r","b"))

plt.show()
def vector_norm(vector):

    squares = [element**2 for element in vector]

    return sum(squares)**0.5



print("||", u, "|| =")

vector_norm(u)
import numpy.linalg as LA

LA.norm(u)
radius = LA.norm(u)

plt.gca().add_artist(plt.Circle((0,0), radius, color="#DDDDDD"))

plot_vector2d(u, color="red")

plt.axis([0, 8.7, 0, 6])

plt.grid()

plt.show()
print(" ", u)

print("+", v)

print("-"*10)

u + v
plot_vector2d(u, color="r")

plot_vector2d(v, color="b")

plot_vector2d(v, origin=u, color="b", linestyle="dotted")

plot_vector2d(u, origin=v, color="r", linestyle="dotted")

plot_vector2d(u+v, color="g")

plt.axis([0, 9, 0, 7])

plt.text(0.7, 3, "u", color="r", fontsize=18)

plt.text(4, 3, "u", color="r", fontsize=18)

plt.text(1.8, 0.2, "v", color="b", fontsize=18)

plt.text(3.1, 5.6, "v", color="b", fontsize=18)

plt.text(2.4, 2.5, "u+v", color="g", fontsize=18)

plt.grid()

plt.show()
t1 = np.array([2, 0.25])

t2 = np.array([2.5, 3.5])

t3 = np.array([1, 2])



x_coords, y_coords = zip(t1, t2, t3, t1)

plt.plot(x_coords, y_coords, "c--", x_coords, y_coords, "co")



plot_vector2d(v, t1, color="r", linestyle=":")

plot_vector2d(v, t2, color="r", linestyle=":")

plot_vector2d(v, t3, color="r", linestyle=":")



t1b = t1 + v

t2b = t2 + v

t3b = t3 + v



x_coords_b, y_coords_b = zip(t1b, t2b, t3b, t1b)

plt.plot(x_coords_b, y_coords_b, "b-", x_coords_b, y_coords_b, "bo")



plt.text(4, 4.2, "v", color="r", fontsize=18)

plt.text(3, 2.3, "v", color="r", fontsize=18)

plt.text(3.5, 0.4, "v", color="r", fontsize=18)



plt.axis([0, 6, 0, 5])

plt.grid()

plt.show()
print("1.5 *", u, "=")



1.5 * u
k = 2.5

t1c = k * t1

t2c = k * t2

t3c = k * t3



plt.plot(x_coords, y_coords, "c--", x_coords, y_coords, "co")



plot_vector2d(t1, color="r")

plot_vector2d(t2, color="r")

plot_vector2d(t3, color="r")



x_coords_c, y_coords_c = zip(t1c, t2c, t3c, t1c)

plt.plot(x_coords_c, y_coords_c, "b-", x_coords_c, y_coords_c, "bo")



plot_vector2d(k * t1, color="b", linestyle=":")

plot_vector2d(k * t2, color="b", linestyle=":")

plot_vector2d(k * t3, color="b", linestyle=":")



plt.axis([0, 9, 0, 9])

plt.grid()

plt.show()
plt.gca().add_artist(plt.Circle((0,0),1,color='c'))

plt.plot(0, 0, "ko")

plot_vector2d(v / LA.norm(v), color="k")

plot_vector2d(v, color="b", linestyle=":")

plt.text(0.3, 0.3, "$\hat{u}$", color="k", fontsize=18)

plt.text(1.5, 0.7, "$u$", color="b", fontsize=18)

plt.axis([-1.5, 5.5, -1.5, 3.5])

plt.grid()

plt.show()
def dot_product(v1, v2):

    return sum(v1i * v2i for v1i, v2i in zip(v1, v2))



dot_product(u, v)
np.dot(u,v)
u.dot(v)
print("  ",u)

print("* ",v, "(NOT a dot product)")

print("-"*10)



u * v
def vector_angle(u, v):

    cos_theta = u.dot(v) / LA.norm(u) / LA.norm(v)

    return np.arccos(np.clip(cos_theta, -1, 1))



theta = vector_angle(u, v)

print("Angle =", theta, "radians")

print("      =", theta * 180 / np.pi, "degrees")
u_normalized = u / LA.norm(u)

proj = v.dot(u_normalized) * u_normalized



plot_vector2d(u, color="r")

plot_vector2d(v, color="b")



plot_vector2d(proj, color="k", linestyle=":")

plt.plot(proj[0], proj[1], "ko")



plt.plot([proj[0], v[0]], [proj[1], v[1]], "b:")



plt.text(1, 2, "$proj_u v$", color="k", fontsize=18)

plt.text(1.8, 0.2, "$v$", color="b", fontsize=18)

plt.text(0.8, 3, "$u$", color="r", fontsize=18)



plt.axis([0, 8, 0, 5.5])

plt.grid()

plt.show()