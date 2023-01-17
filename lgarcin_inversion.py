%matplotlib notebook

from numpy import linspace, logspace, concatenate
from matplotlib.pyplot import plot, show, axis, xlim, ylim, grid, legend
dir = 3 + 1j
vec = 1 + 1j
l = logspace(-10, 1, 1000)
l = concatenate((-l[::-1], l))
droite = dir * l + vec
image = 1. / droite.conjugate()
axis('equal')
xlim([-1, 2])
ylim([-1, 2])
grid()
plot(droite.real, droite.imag, label='droite')
plot(image.real, image.imag, label='image')
legend()
show()
%matplotlib notebook

from numpy import linspace, exp, pi
from matplotlib.pyplot import plot, show, axis, xlim, ylim, grid, legend
centre = 1 + 1j
rayon = 2
cercle = centre + rayon * exp(1j * linspace(-pi, pi, 1000))
image = 1. / cercle.conjugate()
axis('equal')
xlim([-1, 2])
ylim([-2, 4])
grid()
plot(cercle.real, cercle.imag, label='cercle')
plot(image.real, image.imag, label='image')
legend()
show()
%matplotlib notebook

from numpy import linspace, exp, pi, sqrt
from matplotlib.pyplot import plot, show, axis, xlim, ylim, grid, legend
centre = 1 + 1j
rayon = sqrt(2)
cercle = centre + rayon * exp(1j * linspace(-pi + pi/4, pi + pi/4, 1000))
image = 1. / cercle.conjugate()
axis('equal')
xlim([-1, 2])
ylim([-2, 4])
grid()
plot(cercle.real, cercle.imag, label='cercle')
plot(image.real, image.imag, label='image')
legend()
show()
%matplotlib notebook

from numpy import meshgrid, logspace, concatenate
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import figure, show

pt = (0, 0, 1)
dir1 = (1, 0, 0)
dir2 = (0, 1, 0)
l = logspace(-1, 2, 1000)
l = concatenate((-l[::-1], l))
u, v = meshgrid(l, l)
x = dir1[0] * u + dir2[0] * v + pt[0]
y = dir1[1] * u + dir2[1] * v + pt[1]
z = dir1[2] * u + dir2[2] * v + pt[2]
fig = figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')
ax.plot_surface(x, y, z)

fig = figure()
ax = fig.gca(projection='3d')
ax.set_aspect('equal')
n = x ** 2 + y ** 2 + z ** 2
xx = x / n
yy = y / n
zz = z / n
ax.plot_surface(xx, yy, zz)
show()
