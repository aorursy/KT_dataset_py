from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d
%matplotlib notebook
# Constants related to an ideal gas
R = 0.082 # Liter-atm/mol-K
V = 20 # Liters
T = 250 # Kelvin
n = 1 # moles

# van der Waals constants
a = 2.00
b = 0.04

# Get pressure of an ideal gas
P = n*R*T/V
print("Pressure of ideal gas = ", P, "atm")

# Get pressure of a vdw gas
Pvdw = n*R*T/(V-n*b)-a*n**2/V**2
print("Pressure of vdw = ", Pvdw, "atm")

# Get percent error (using Eq. 3 in the handout)
Error = (P-Pvdw)/Pvdw*100
print("%Error = ", Error)
# Generate a range of volumes
V_array = linspace(5,40)
print("There are", shape(V_array), "points in V") # shape tells you the length of the array
print(V_array)

# Get pressure of an ideal gas
P = n*R*T/V_array
print("There are", P.shape, "points in P")

# Open up a figure window
figure()

# Graph P(V)
plot(V_array,P) # Plot the ideal gas Boyle isotherm
xlabel('V (L)') # Label the x axis
ylabel('P (atm)') # Label the y axis

# Get pressure of a vdw gas

# Open up a figure window

# Graph Pvdw(V)

# Get the %error

# Print the %error

V_array = linspace(5,40)
T_array = linspace(200,300)
V_grid, T_grid = meshgrid(V_array,T_array) # Make a grid covering every V & T combination 
print("There are", shape(V_grid), "points in V")
print("There are", shape(T_grid), "points in T")
# Get pressure grid of ideal gas for every point on the grid
P = n*R*T_grid/V_grid
print("There are", shape(P), "points in P")
print(P)

# Open up a 3d figure window
ax = figure().gca(projection='3d') # Set up a three dimensional graphics window 

# Graph the pressure
ax.plot_surface(V_grid, T_grid, P) # Make the mesh plot P(V,T)
ax.set_xlabel('V (L)') # Label axes
ax.set_ylabel('T (K)')
ax.set_zlabel('P (atm)')
# Get pressure grid of van der Waals gas (Pvdw) for every point on the grid

# Open up a 3d figure window

# Graph the pressure

# Calculate the %error

# Open up a 3d figure window

# Graph the error

