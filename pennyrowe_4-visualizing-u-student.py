from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d
%matplotlib notebook
# Load the thermodynamic state space
V = loadtxt('../input/data-for-visualizing-u/V.txt') # In m^3
T = loadtxt('../input/data-for-visualizing-u/T.txt') # In K

# Load the state functions (In kJ/mol ... rows are for volume, columns are for temperature)
Ugrid1 = loadtxt('../input/data-for-visualizing-u/Ugrid1.txt')  
Ugrid2 = loadtxt('../input/data-for-visualizing-u/Ugrid2.txt') 
Ugrid3 = loadtxt('../input/data-for-visualizing-u/Ugrid3.txt')
# Open a figure
figure()

# Extract the isochore of the first gas and plot it, 
UofT1 = Ugrid1[1,:]
plot(T, UofT1, label='gas 1')

# Extract the isochore of the second and plot it
UofT2 = Ugrid2[1,:]
plot(T, UofT2, label='gas 2')

# Extract the isochore of the third and plot it
UofT3 = Ugrid3[1,:]
plot(T, UofT3, label='gas 3')

# Ask for legends
legend()

#Add axes labels

# Open a figure
figure()

# Extract the isotherm of the first gas and plot it, 
UofV1 = Ugrid1[:,1]
plot(T, UofV1, label='gas 1')

# Extract the isotherm of the second and plot it
UofV2 = Ugrid2[:,1]
plot(T, UofV2, label='gas 2')

# Extract the isotherm of the third and plot it
UofV3 = Ugrid3[:,1]
plot(T, UofV3, label='gas 3')

# Ask for legends


# Add axes labels

Tgrid, Vgrid = meshgrid(T,V)
# Open up a 3d figure window
ax = figure().gca(projection='3d') # Set up a three dimensional graphics window 

# Plot surface 1
ax.plot_surface(Tgrid, Vgrid*1e3, Ugrid1, color='blue') # Make the mesh plot
ax.set_ylabel('V (L)') # Label axes
ax.set_xlabel('T (K)')
ax.set_zlabel('U (kJ/mol)')

# Plot surface 2
ax.plot_surface(Tgrid, Vgrid*1e3, Ugrid2, color='red') # Make the mesh plot
ax.set_ylabel('V (L)') # Label axes
ax.set_xlabel('T (K)')
ax.set_zlabel('U (kJ/mol)')

# Plot surface 3
ax.plot_surface(Tgrid, Vgrid*1e3, Ugrid3, color='green') # Make the mesh plot
ax.set_ylabel('V (L)') # Label axes
ax.set_xlabel('T (K)')
ax.set_zlabel('U (kJ/mol)')
# Open up a 3d figure window
ax = figure().gca(projection='3d') # Set up a three dimensional graphics window 

# Calculate f 


# Display it as a thermodynamic surface

