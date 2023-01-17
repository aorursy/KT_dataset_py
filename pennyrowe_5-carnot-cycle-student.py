from numpy import *
from matplotlib.pyplot import *
%matplotlib notebook
def extract(V,P,Vstart,Vstop):
    if Vstop > Vstart:
        index = argwhere((V>Vstart) & (V<Vstop))
    else:
        index = argwhere((V<Vstart) & (V>Vstop))
    return squeeze(V[index]), squeeze(P[index])
# Gas properties
R = 8.314
n = 1
C_V = 3./2*R*n # This is a monatomic ideal gas
c = C_V/(R*n) # This is the "reduced heat capacity" for adiabatic reversible expansion
print('c =', c)
gamma = 1 + 1/c
print('gamma =', gamma)

# For the Carnot cycle, define the hot/cold reservoirs and the volume arrays that span the four legs
T_hot = 400.0
T_cold = 300.0
V_leg1 = linspace(0.01, 0.10, 200)
V_leg2 = linspace(0.01, 0.10, 200)
V_leg3 = linspace(0.10, 0.01, 200) # Going backward because we're compressing on these legs
V_leg4 = linspace(0.10, 0.01, 200) # Going backward because we're compressing on these legs
# Hot Boyle isotherm
P_leg1 = R*T_hot/V_leg1

# Graph it
figure()
plot(V_leg1,P_leg1,'r',label='hot isothermal expansion')
grid(True)
legend()
# Choose a point on the hot isotherm that we want the adiabat to intersect
V_on_hot_isotherm = 0.04
P_on_hot_isotherm = R*T_hot/V_on_hot_isotherm

# Calculate an adiabat that crosses the hot isotherm at that point
P_leg2 = P_on_hot_isotherm*(V_on_hot_isotherm/V_leg2)**gamma # Formula for adiabatic reversible expansion

# Graph them together
figure()
plot(V_leg1,P_leg1,'r',label='hot isothermal expansion')
plot(V_leg2,P_leg2,'k--',label='adiabatic expansion')
grid(True)
legend()
# Calculate the cold Boyle isotherm (call it "P_leg3")
P_leg3 = R*T_cold/V_leg3

# Graph all three so far

# Choose a point on the cold isotherm that we want the adiabat to intersect
V_on_cold_isotherm = 0.04
P_on_cold_isotherm = R*T_cold/V_on_cold_isotherm

# Calculate an adiabat that crosses the cold isotherm at that point (call it "P_leg4")
P_leg4 = P_on_cold_isotherm*(V_on_cold_isotherm/V_leg4)**gamma # Formula for adiabatic reversible expansion

# Graph legs 1-4 together

# Specify the volumes that define the intersections of the Carnot cycle
VA = 0.03
VB = 0.05
VC = 0.07
VD = 0.05

# Extract the Carnot range of the first leg
V_leg1_Carnot, P_leg1_Carnot = extract(V_leg1,P_leg1,VA,VB)

# Extract the Carnot range of the second leg
V_leg2_Carnot, P_leg2_Carnot = extract(V_leg2,P_leg2,VB,VC)

# Extract the Carnot range of the third leg ... you'll need to "uncomment" this line (remove hash-tags and space)
V_leg3_Carnot, P_leg3_Carnot = extract(V_leg3,P_leg3,VC,VD)

# Extract the Carnot range of the fourth leg ... you'll need to "uncomment" this line 
V_leg4_Carnot, P_leg4_Carnot = extract(V_leg4,P_leg4,VD,VA)

# Graph them all together ... you'll need to "uncomment" these lines 
figure()
plot(V_leg1_Carnot,P_leg1_Carnot,'r', label='hot isothermal expansion')
plot(V_leg2_Carnot,P_leg2_Carnot,'k--', label='adiabatic expansion')
plot(V_leg3_Carnot,P_leg3_Carnot,'b', label='cold isothermal compression')
plot(V_leg4_Carnot,P_leg4_Carnot,'k-.',label='adiabatic compression')
grid(True)
legend()
# Calculate and print the volume ratios (for comparison)

# Get the work of each leg (w_leg1, w_leg2, etc)
w_leg1 = -trapz(P_leg1_Carnot,V_leg1_Carnot); print('w1 =', w_leg1)
w_leg2 = -trapz(P_leg2_Carnot,V_leg2_Carnot); print('w2 =', w_leg2)
w_leg3 = -trapz(P_leg3_Carnot,V_leg3_Carnot); print('w3 =', w_leg3)
w_leg4 = -trapz(P_leg4_Carnot,V_leg4_Carnot); print('w4 =', w_leg4)
# Calculate and print the total work
# Calculate & print q_hot from the work done on that leg (Eq. 8)


# Calculate & print the efficiency of the heat engine based on the heat and work (Eq. 9)
# Note that this is the numerical result for an ideal gas (not a real gas). 

# Calculate & print the theoretical q_hot from Eq. 10

# Now calculate & print the theoretical efficiency (based on the temperatures of the reservoirs) 
# from Eq. 11

