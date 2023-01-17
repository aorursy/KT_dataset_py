from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import axes3d
%matplotlib notebook
R = 8.314e-3 # kJ/mol-K
T = 310 # Physiological temperature
#Define concentration gradient with pH1 and pH2 
pH1 = 8 
pH2 = 7

#calculate H+ concentrations, c1 and c2, corresponding to pH1 and pH2

#Define eps

# Make a linear path along eps that starts at c1 and ends at c2 as in Eq 3


# Graph c(eps)
figure()
plot(eps,c)
xlabel('eps')
ylabel('concentration of H+ according to path 1')

# Calculate dc/deps


# Form the integrand (including dc/deps)


# Integrate using trapz and print result

print("Reversible work for path 1 is", w)

# Make a different path along eps that starts at c1 and ends at c2

# Graph c(eps)



# Calculate dc/deps

# Form the integrand (including dc/deps)


# Integrate and print result

# Evaluating the analytical result

