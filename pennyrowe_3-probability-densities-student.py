# This imports various libraries.
from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d
%matplotlib notebook
# This part explores the temperature dependence of fb(v) 
R = 8.314
n = 1
M = 0.028

# Calculate a grid of velocities and temperatures
vx = linspace(-200, 200)
T = linspace(50, 500)
vxgrid, Tgrid = meshgrid(vx, T)

# Get the probability density for every point on the grid
D = M / (2 * R * Tgrid)
N_vx = sqrt(M / (2 * pi * R * Tgrid))
f_vx = N_vx * exp(-D * vxgrid**2)

# Set up a figure for 3d graphics
ax = figure().gca(projection='3d')

# Make the mesh plot; the "stride" parameters improve the appearance
ax.plot_surface(vxgrid, Tgrid, f_vx, rstride=2, cstride=2) 
ax.set_xlabel('vx (m/s)') # Label axes
ax.set_ylabel('T (K)')
ax.set_zlabel('f_v_x (s/m)')
# An array of speeds and temperatures
v = linspace(0, 200)
T = linspace(50, 500)
vgrid, Tgrid = meshgrid(v, T)

# Get the probability density for every point on the grid
D = M / (2 * R * Tgrid)
N_v = sqrt(2) * M**1.5 * R**(-1.5) * Tgrid**(-1.5) * pi**(-0.5)
f_v = N_v * vgrid**2 * exp(-D * vgrid**2)
# Set up a figure for 3d graphics

# Make the mesh plot

# Moments for Velocity Component

# Lay out an array of velocities and their probability densities at a single temperature
T = 50
D = M / (2*R*T)
N_vx = sqrt(M / (2*pi*R*T))
vx = linspace(-200, 200)
f_vx = N_vx * exp(-D * vx**2)

# Plot the integrand for the first moment, and calculate the moment using the trapezoidal rule
figure()            # Set up a graphics window 
plot(vx, f_vx * vx)  # Plot the integrand
grid(True)          # Put a grid on the plot
xlabel('v_x (m/s)') # Label the x axis
ylabel('f_vx * v_x') # Label the y axis
print('The mean of v_x is', trapz(f_vx * vx, vx))

# Do the same for the second moment 
figure()
plot(vx, f_vx * vx**2)
grid(True)
xlabel('v_x (m/s)') # Label the x axis
ylabel('f_vx * v_x^2') # Label the y axis
print('The mean of vx^2 is', trapz(f_vx * vx**2, vx))

# Do the same for the third moment 

# Moments for Speeds

# Lay out an array of velocities and their probability densities at a single temperature
T = 50
D = M / (2*R*T)
N_v = sqrt(2) * M**1.5 * R**(-1.5) * T**(-1.5) * pi**(-0.5)
v = linspace(0, 200)
f_v = N_v * v**2 * exp(-D*v**2)

# Plot the integrand for the first moment, and calculate the moment (called "c-bar") using the trapezoidal rule
figure() # Set up a graphics window 
plot(v, f_v * v)
grid(True)
cbar = trapz(f_v * v, v)
print(cbar)

# Do the same for the second moment and its square root (called "c")

# Do the same for the third moment and its cubed root (called "ctilde")
