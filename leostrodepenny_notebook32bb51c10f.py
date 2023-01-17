import numpy as np

%matplotlib notebook

import matplotlib.pyplot as plt
# ------------ INITIALISATION ---------------------



# --- INPUT ---



q_m_ratio = 1       # charge-to-mass ratio

b0 = 1              # magnetic field (homogeneous, in z direction) 



v =  10             # enter the correct initial velocity.

phi =  0            # enter the correct phase



rxc = 0             # enter the correct x-coordinate of the centre of the circle

ryc = 0             # enter the correct y-coordinate of the centre of the circle



start_time = 0      # enter the start time of the propagation

N_loops = 2         # enter number of loops the particle should complete 

steps_rev = 1000    # enter the number of time steps per revolution 



# --- calculate values from input variables



omega = b0*q_m_ratio                           # calculate the cyclotron frequency (from the input variables)



period_osc = 2*np.pi/omega                    # calculate the period of the oscillation (from the input variables)



end_time = N_loops*period_osc                        # calculate the end time of the propagation (from the input variables)



time_steps =  steps_rev*N_loops                     # calculate the number of time steps (from the input variables)



# --- defininition of time array for propagation 



step = (end_time-start_time)/(time_steps-1)

time = np.linspace(start_time,end_time,time_steps)
#ANALYTICAL SOLUTION of the differential equations

exact_position_x = v/omega*np.sin(omega*time - phi) + rxc   # multiplication with numpy array time

exact_position_y = v/omega*np.cos(omega*time - phi) + ryc   # yields a numpy array

exact_velocity_x = v*np.cos(omega*time - phi)

exact_velocity_y = -v*np.sin(omega*time - phi)
#Plot of exact solutions to check that they make sense.

plt.figure(1)

plt.plot(time,exact_position_x,linewidth=3,label='x position')

plt.plot(time,exact_position_y,linewidth=3,label='y position')

plt.grid(True)

plt.legend()

plt.title(f"x and y component of particle's position vs. time")

plt.xlabel(f"time [s]") 

plt.ylabel(f"position [m]") 

plt.show()
# If we plot r_y(t) against r_x(t), the result should be a circle.

plt.figure(2)

plt.plot(exact_position_x,exact_position_y)

plt.axis('equal') #scale of x and y axes is the same to show that motion is along a circle

plt.title(f"cyclotron motion of particle in x-y plane")

plt.grid(True)

plt.xlabel(f"x position [m]") 

plt.ylabel(f"y position [m]") 

plt.show()
# initialisation of position and velocity arrays, setting everything to zero

position_x = np.zeros(time_steps)

position_y = np.zeros(time_steps)

velocity_x = np.zeros(time_steps)

velocity_y = np.zeros(time_steps)
# --- Definition of the derivative of y(t), F(y(t)) ---

#

# Input:  yin :  vector y (numpy array)

#         omega: cyclotron frequency

# Output: derivative of vector yin, F(yin) (numpy array)



def der(yin, omega):                                              # no explicit time dependence

    return np.array([yin[2],yin[3],omega*yin[3],-omega*yin[2]])
# --- Implementation of Euler method

# Input: yin - initial vector of position and velocity

#        omega - cylcotron frequency

#        step - time step

#

# Output: yout - propagated vector of position and velocity

def euler(yin,omega,step):

    return yin+step*np.array([yin[2],yin[3],omega*yin[3],-omega*yin[2]])



    #complete the function definition here

    
#--- Input of initial conditions



position_x[0] = 0.0        # initial position

position_y[0] = v/omega  

velocity_x[0] = v          # initial velocity

velocity_y[0] = 0.0  



yin = np.zeros(4)          # initialisation of yin



yin[0] = position_x[0]     # start with initial conditions 

yin[1] = position_y[0]

yin[2] = velocity_x[0]

yin[3] = velocity_y[0]



#--- Propagation



for ii in range(1,time_steps):

    yin = euler(yin,omega,step)                    # calculation of yout (which immediately overwrites the old yin!)

    

    position_x[ii] = yin[0]                        # save the new position and velocity components

    position_y[ii] = yin[1]

    velocity_x[ii] = yin[2]

    velocity_y[ii] = yin[3]
plt.figure(3)

plt.plot(exact_position_x,exact_position_y,linewidth=3,label='exact position')

plt.plot(position_x,position_y,linewidth=1,label='numerical position')

plt.axis('equal')

plt.legend()

plt.title(f"Comparison for analytical and numerical solution")

plt.xlabel(f"x position [m]") 

plt.ylabel(f"y position [m]") 

plt.show()
# y-t plot of cyclotron motion 

# comparison analytical vs. numerical solution



plt.figure(4)

plt.plot(time,position_y,'x',label='numerical position')

plt.plot(time,exact_position_y,linewidth=1,label='Analytical position')

plt.axis([0,N_loops*2*np.pi/omega,-1*(position_y[0]+0.1),position_y[0]+0.1])

plt.title(f"Comparison analytical vs numerical result of y(t)")

plt.xlabel(f"time [s]") 

plt.ylabel(f"y position [m]") 

plt.legend

plt.show()
import numpy as np

%matplotlib notebook

import matplotlib.pyplot as plt



# ------------ INITIALISATION ---------------------



# --- INPUT ---



print("Part a")





q_m_ratio = 1       # charge-to-mass ratio

b0 = 1              # magnetic field (homogeneous, in z direction) 



v =  10             # initial velocity.

phi =  0            # phase



rxc = 0             # x-coordinate of the centre of the circle

ryc = 0             # y-coordinate of the centre of the circle



start_time = 0      # start time of the propagation

N_loops = 8         # number of loops the particle should complete - set to 8

steps_rev = 1000    # number of time steps per revolution 



# --- calculate values from input variables



omega = b0*q_m_ratio                           # cyclotron frequency (from the input variables)



period_osc = 2*np.pi/omega                    # period of the oscillation (from the input variables)



end_time = N_loops*period_osc                        # end time of the propagation (from the input variables)



time_steps =  steps_rev*N_loops                     # number of time steps (from the input variables)



# --- defininition of time array for propagation 



step = (end_time-start_time)/(time_steps-1)

time = np.linspace(start_time,end_time,time_steps)



#ANALYTICAL SOLUTION of the differential equations

exact_position_x = v/omega*np.sin(omega*time - phi) + rxc   # multiplication with numpy array time

exact_position_y = v/omega*np.cos(omega*time - phi) + ryc   # yields a numpy array

exact_velocity_x = v*np.cos(omega*time - phi)

exact_velocity_y = -v*np.sin(omega*time - phi)



#Plot of exact solutions to check that they make sense.

plt.figure(1)

plt.plot(time,exact_position_x,linewidth=3,label='x position')

plt.plot(time,exact_position_y,linewidth=3,label='y position')

plt.grid(True)

plt.legend()

plt.title(f"x and y component of particle's position vs. time")

plt.xlabel(f"time [s]") 

plt.ylabel(f"position [m]") 

plt.show()



# If we plot r_y(t) against r_x(t), the result should be a circle.

plt.figure(2)

plt.plot(exact_position_x,exact_position_y)

plt.axis('equal') #scale of x and y axes is the same to show that motion is along a circle

plt.title(f"cyclotron motion of particle in x-y plane")

plt.grid(True)

plt.xlabel(f"x position [m]") 

plt.ylabel(f"y position [m]") 

plt.show()



# initialisation of position and velocity arrays, setting everything to zero

position_x = np.zeros(time_steps)

position_y = np.zeros(time_steps)

velocity_x = np.zeros(time_steps)

velocity_y = np.zeros(time_steps)



# --- Definition of the derivative of y(t), F(y(t)) ---

#

# Input:  yin :  vector y (numpy array)

#         omega: cyclotron frequency

# Output: derivative of vector yin, F(yin) (numpy array)



def der(yin, omega):                                              # no explicit time dependence

    return np.array([yin[2],yin[3],omega*yin[3],-omega*yin[2]])



# --- Implementation of Euler method

# Input: yin - initial vector of position and velocity

#        omega - cylcotron frequency

#        step - time step

#

# Output: yout - propagated vector of position and velocity

def euler(yin,omega,step):

    return yin+step*np.array([yin[2],yin[3],omega*yin[3],-omega*yin[2]])





    

#--- Input of initial conditions



position_x[0] = 0.0        # initial position

position_y[0] = v/omega  

velocity_x[0] = v          # initial velocity

velocity_y[0] = 0.0  



yin = np.zeros(4)          # initialisation of yin



yin[0] = position_x[0]     # start with initial conditions 

yin[1] = position_y[0]

yin[2] = velocity_x[0]

yin[3] = velocity_y[0]



#--- Propagation



for ii in range(1,time_steps):

    yin = euler(yin,omega,step)                    # calculation of yout (which immediately overwrites the old yin!)

    

    position_x[ii] = yin[0]                        # save the new position and velocity components

    position_y[ii] = yin[1]

    velocity_x[ii] = yin[2]

    velocity_y[ii] = yin[3]

    

plt.figure(3)

plt.plot(exact_position_x,exact_position_y,linewidth=3,label='exact position')

plt.plot(position_x,position_y,linewidth=1,label='numerical position')

plt.axis('equal')

plt.legend()

plt.title(f"Comparison for analytical and numerical solution")

plt.xlabel(f"x position [m]") 

plt.ylabel(f"y position [m]") 

plt.show()



print("A line forming a spiral is observed in Fig.3. As more loops are completed by the partical,", 

      "the numerically calculated position (shown by the orange line) of the partical gets further", 

      "away from its theoretical position (the blue line). This means the accuracy of the numerical", 

      "result decreases as the number of loops completed increases.")





plt.figure(4)

plt.plot(time,position_y,'x',label='numerical position')

plt.plot(time,exact_position_y,linewidth=1,label='Analytical position')

plt.axis([0,N_loops*2*np.pi/omega,-1*(position_y[0]+0.1),position_y[0]+0.1])

plt.title(f"Comparison analytical vs numerical result of y(t)")

plt.xlabel(f"time [s]") 

plt.ylabel(f"y position [m]") 

plt.legend

plt.show()



print("As time progresses, the amplitude of the particle in the y-direction increases when calculated", 

      "numerically, and stays constant when calculated analytically. This means that as time progresses", 

      "the numerical calculation becomes increasingly inacurate.")



print("\n")

print("Part b")

print("A way to reduce error in the analytical solution is to increase the number of increments the", 

      "euler method is used (i.e : increase the number of measured time increments). In next plots, the", 

      "number of time increments per revolution has been increased from 1000 to 10000;")



# ------------ INITIALISATION ---------------------



# --- INPUT ---







q_m_ratio = 1       # charge-to-mass ratio

b0 = 1              # magnetic field (homogeneous, in z direction) 



v =  10             # initial velocity.

phi =  0            # phase



rxc = 0             # x-coordinate of the centre of the circle

ryc = 0             # y-coordinate of the centre of the circle



start_time = 0      # start time of the propagation

N_loops = 8         # number of loops the particle should complete - set to 8

steps_rev = 10000    # number of time steps per revolution - set to 10000 (10x original)



# --- calculate values from input variables



omega = b0*q_m_ratio                           # cyclotron frequency (from the input variables)



period_osc = 2*np.pi/omega                    # period of the oscillation (from the input variables)



end_time = N_loops*period_osc                        # end time of the propagation (from the input variables)



time_steps =  steps_rev*N_loops                     # number of time steps (from the input variables)



# --- defininition of time array for propagation 



step = (end_time-start_time)/(time_steps-1)

time = np.linspace(start_time,end_time,time_steps)



#ANALYTICAL SOLUTION of the differential equations

exact_position_x = v/omega*np.sin(omega*time - phi) + rxc   # multiplication with numpy array time

exact_position_y = v/omega*np.cos(omega*time - phi) + ryc   # yields a numpy array

exact_velocity_x = v*np.cos(omega*time - phi)

exact_velocity_y = -v*np.sin(omega*time - phi)



#Plot of exact solutions to check that they make sense.

plt.figure(5)

plt.plot(time,exact_position_x,linewidth=3,label='x position')

plt.plot(time,exact_position_y,linewidth=3,label='y position')

plt.grid(True)

plt.legend()

plt.title(f"x and y component of particle's position vs. time")

plt.xlabel(f"time [s]") 

plt.ylabel(f"position [m]") 

plt.show()





# If we plot r_y(t) against r_x(t), the result should be a circle.

plt.figure(6)

plt.plot(exact_position_x,exact_position_y)

plt.axis('equal') #scale of x and y axes is the same to show that motion is along a circle

plt.title(f"cyclotron motion of particle in x-y plane")

plt.grid(True)

plt.xlabel(f"x position [m]") 

plt.ylabel(f"y position [m]") 

plt.show()



print("Plots 5 and 6 can be seen to have been unchanged as a result of increasing time increments.")





# initialisation of position and velocity arrays, setting everything to zero

position_x = np.zeros(time_steps)

position_y = np.zeros(time_steps)

velocity_x = np.zeros(time_steps)

velocity_y = np.zeros(time_steps)



# --- Definition of the derivative of y(t), F(y(t)) ---

#

# Input:  yin :  vector y (numpy array)

#         omega: cyclotron frequency

# Output: derivative of vector yin, F(yin) (numpy array)



def der(yin, omega):                                              # no explicit time dependence

    return np.array([yin[2],yin[3],omega*yin[3],-omega*yin[2]])



# --- Implementation of Euler method

# Input: yin - initial vector of position and velocity

#        omega - cylcotron frequency

#        step - time step

#

# Output: yout - propagated vector of position and velocity

def euler(yin,omega,step):

    return yin+step*np.array([yin[2],yin[3],omega*yin[3],-omega*yin[2]])





    

#--- Input of initial conditions



position_x[0] = 0.0        # initial position

position_y[0] = v/omega  

velocity_x[0] = v          # initial velocity

velocity_y[0] = 0.0  



yin = np.zeros(4)          # initialisation of yin



yin[0] = position_x[0]     # start with initial conditions 

yin[1] = position_y[0]

yin[2] = velocity_x[0]

yin[3] = velocity_y[0]



#--- Propagation



for ii in range(1,time_steps):

    yin = euler(yin,omega,step)                    # calculation of yout (which immediately overwrites the old yin!)

    

    position_x[ii] = yin[0]                        # save the new position and velocity components

    position_y[ii] = yin[1]

    velocity_x[ii] = yin[2]

    velocity_y[ii] = yin[3]

    

plt.figure(7)

plt.plot(exact_position_x,exact_position_y,linewidth=3,label='exact position')

plt.plot(position_x,position_y,linewidth=1,label='numerical position')

plt.axis('equal')

plt.legend()

plt.title(f"Comparison for analytical and numerical solution")

plt.xlabel(f"x position [m]") 

plt.ylabel(f"y position [m]") 

plt.show()



print("The variation between the numerically and analytically calculated positions have reduced ",

      "significantly in Fig.7 when compared to Fig.3. As the number of loops increases the deviation",

      "between the two lines is observed to be minimal.")



# y-t plot of cyclotron motion 

# comparison analytical vs. numerical solution



plt.figure(8)

plt.plot(time,position_y,'x',label='numerical position')

plt.plot(time,exact_position_y,linewidth=1,label='Analytical position')

plt.axis([0,N_loops*2*np.pi/omega,-1*(position_y[0]+0.1),position_y[0]+0.1])

plt.title(f"Comparison analytical vs numerical result of y(t)")

plt.xlabel(f"time [s]") 

plt.ylabel(f"y position [m]") 

plt.legend

plt.show()



print("The variation of amplitude in the y direction can also be seen to have decreased significantly in this plot.")



print("Part c")

print("\n","In this section, the effect of speed of the particle on its radius will be investigated.","\n","The effect",

     "of doubling the velocity (from 10-20) is shown:")



v =  20             # initial velocity. - doubled



time = np.linspace(start_time,end_time,time_steps)



#ANALYTICAL SOLUTION of the differential equations

exact_position_x = v/omega*np.sin(omega*time - phi) + rxc   # multiplication with numpy array time

exact_position_y = v/omega*np.cos(omega*time - phi) + ryc   # yields a numpy array

exact_velocity_x = v*np.cos(omega*time - phi)

exact_velocity_y = -v*np.sin(omega*time - phi)





# If we plot r_y(t) against r_x(t), the result should be a circle.

plt.figure(9)

plt.plot(exact_position_x,exact_position_y)

plt.axis('equal') #scale of x and y axes is the same to show that motion is along a circle

plt.title(f"cyclotron motion of particle in x-y plane")

plt.grid(True)

plt.xlabel(f"x position [m]") 

plt.ylabel(f"y position [m]") 

plt.show()



print("As we can see from Fig.9, the radius has doubled to 20.","\n","The effect",

     "of halving the velocity (from 10-5) is shown:")



v =  5             # initial velocity. - halved



time = np.linspace(start_time,end_time,time_steps)



#ANALYTICAL SOLUTION of the differential equations

exact_position_x = v/omega*np.sin(omega*time - phi) + rxc   # multiplication with numpy array time

exact_position_y = v/omega*np.cos(omega*time - phi) + ryc   # yields a numpy array

exact_velocity_x = v*np.cos(omega*time - phi)

exact_velocity_y = -v*np.sin(omega*time - phi)





# If we plot r_y(t) against r_x(t), the result should be a circle.

plt.figure(10)

plt.plot(exact_position_x,exact_position_y)

plt.axis('equal') #scale of x and y axes is the same to show that motion is along a circle

plt.title(f"cyclotron motion of particle in x-y plane")

plt.grid(True)

plt.xlabel(f"x position [m]") 

plt.ylabel(f"y position [m]") 

plt.show()





print("As we can see from Fig.10, the radius has halved to 5.")



print("\n","These two results are as expected. As shown above in the derived equations: r(x) = (v/w)*sin(wt) and r(y) = (v/w)*cos(wt).",

     "This shows that the maximum displacements in both the x and y directions are proportional to the speed of the particle, v.",

     "This means that if v were to double, so would the maximum displacements in both the x and y direction (both of these values are the",

     "same, and are also equal to the radius of the path of the particle).")

# --- Definition of the derivative of y(t) ---

# Input: time: time at which the derivative is evaluated (not needed here, because we have no explicit time-dependence)

#        yin : array for vector y (containing 4 components)

#        omega: cyclotron frequency

# Output: yout: propagated vector y



def der_rk(yin, omega):

    return np.array([yin[2],yin[3],omega*yin[3],-omega*yin[2]])
# --- Implementation of the Runge Kutta method



def runge_kutta(yin,omega,step):

    k1 = step*der_rk(yin,omega)  

    k2 = step*der_rk(yin+k1/2,omega)

    k3 = step*der_rk(yin+k2/2,omega)

    k4 = step*der_rk(yin+k3,omega)

    yout = yin+k1/6.0+k2/3.0+k3/3.0+k4/6.0

    return yout
# initialisation of position and velocity arrays for Runge Kutta method, setting everything to zero

position_rk_x = np.zeros(time_steps)

position_rk_y = np.zeros(time_steps)

velocity_rk_x = np.zeros(time_steps)

velocity_rk_y = np.zeros(time_steps)
for ii in range(1,time_steps):

    yin = runge_kutta(yin,omega_f,step,time[ii])   # calculation of yout (which immediately overwrites the old yin!)

    

    position_x[ii] = yin[0]                        # save the new position and velocity components

    position_y[ii] = yin[1]

    velocity_x[ii] = yin[2]

    velocity_y[ii] = yin[3]
# ------------ INITIALISATION ---------------------



# --- INPUT ---



q_m_ratio = 1.0    # charge-to-mass ratio

b0 = 1.0           # magnetic field (homogeneous, in z direction) 



v = 10.0           #enter the correct initial velocity.

phi = 0.0          #enter the correct phase



rxc = 0.0          #enter the correct x-coordinate of the centre of the circle

ryc = 0.0          #enter the correct y-coordinate of the centre of the circle



start_time = 0.0   # enter the start time of the propagation

N_loops = 2        # enter number of loops the particle should complete 

steps_rev = 1000     # enter the number of time steps per revolution 



# --- calculate values from input variables



omega_f = q_m_ratio * b0           # calculate the cyclotron frequency (from the input variables)



period_osc = 2.0 * np.pi/omega_f   # calculate the period of the oscillation (from the input variables)



end_time = N_loops * period_osc    # calculate the end time of the propagation (from the input variables)



time_steps = N_loops * steps_rev   # calculate the number of time steps (from the input variables)



# defininition of time array for propagation 



step = (end_time-start_time)/(time_steps-1)

time = np.linspace(start_time,end_time,time_steps)



#ANALYTICAL SOLUTION of the differential equations

exact_position_x = v/omega_f*np.sin(omega_f*time - phi) + rxc   # multiplication with numpy array time

exact_position_y = v/omega_f*np.cos(omega_f*time - phi) + ryc   # yields a numpy array

exact_velocity_x = v*np.cos(omega_f*time - phi)

exact_velocity_y = -v*np.sin(omega_f*time - phi)





radius = np.sqrt(exact_position_x[0]**2+exact_position_y[0]**2) # exact radius 





#NUMERICAL SOLUTION of the differential equations - Euler method



# initialisation of position and velocity arrays, setting everything to zero

position_x = np.zeros(time_steps)

position_y = np.zeros(time_steps)

velocity_x = np.zeros(time_steps)

velocity_y = np.zeros(time_steps)



diff_radius = np.zeros(time_steps)

diff_velocity = np.zeros(time_steps)



#Input of initial conditions

position_x[0] = 0.0        # Make sure you enter the correct initial conditions.

position_y[0] = v/omega_f  # Enter them in terms of the speed of the particle and the frequency

velocity_x[0] = v          

velocity_y[0] = 0.0  



yin = np.zeros(4)          #initialisation of yin



yin[0] = position_x[0]     #start with initial conditions 

yin[1] = position_y[0]

yin[2] = velocity_x[0]

yin[3] = velocity_y[0]





#numerical solution

for ii in range(1,time_steps):

    yin = euler(yin,omega_f,step)                  # calculation of yout (which immediately overwrites the old yin!)

    

    position_x[ii] = yin[0]                        # save the new position and velocity components

    position_y[ii] = yin[1]

    velocity_x[ii] = yin[2]

    velocity_y[ii] = yin[3]

    

    #derivation of numerical results from exact ones: radius and velocity

    

    diff_radius[ii] = np.abs(np.sqrt(position_x[ii]**2 + position_y[ii]**2) - radius)

    diff_velocity[ii] = np.abs(np.sqrt(velocity_x[ii]**2 + velocity_y[ii]**2) - v)

    

plt.figure(5)

plt.plot(time,diff_radius)

plt.title(f"Derivation of radius from exact one over time")

plt.xlabel(f"time [s]") 

plt.ylabel(f"difference in radii [m]") 

plt.show()



plt.figure(6)

plt.plot(time,diff_velocity)

plt.title(f"Derivation of velocity from exact one over time")

plt.xlabel(f"time [s]") 

plt.ylabel(f"difference in velocities [m/s]") 

plt.show()

time_steps=10000

start_time=0.0

end_time=40.0

step=(end_time-start_time)/(time_steps-1)

time=np.linspace(start_time,end_time,time_steps)

position_x=np.zeros(time_steps)

position_y=np.zeros(time_steps)

position_z=np.zeros(time_steps)
def der(time,yin,omega_f):

    return np.array([10*(yin[1]-yin[0]),yin[0]*(28-yin[2])-yin[1],yin[0]*yin[1]-8/3*yin[2]])
position_x[0]=1.0  #make sure you enter the correct initial conditions.

position_y[0]=1.0  #make sure you enter the correct initial conditions.

position_z[0]=1.0  #make sure you enter the correct initial conditions.

yin=np.zeros(3)

yin[0]=position_x[0]

yin[1]=position_y[0]

yin[2]=velocity_x[0]
for ii in range(1,time_steps):

    yin=runge_kutta(yin,1.0,step,time[ii])

    position_x[ii]=yin[0]

    position_y[ii]=yin[1]

    position_z[ii]=yin[2]

   
plt.figure(7)

plt.plot(position_x,position_z,linewidth=1,label='numerical position')

#plt.axis('equal')

plt.legend()

plt.show()
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(8)

ax = fig.gca(projection='3d')

ax.plot(position_x,position_y,position_z,linewidth=1,label='numerical solution')

ax.legend()

plt.show()
from scipy.integrate import odeint
def lorenz(x,t,A,B,C):

    return np.array([A*(x[1]-x[0]),x[0]*(B-x[2])-x[1],x[0]*x[1]-C*x[2]])

xout=odeint(lorenz,[1,1,1],time,args=(10,28,8/3))

xt=np.transpose(xout)

xx=xt[0]

yy=xt[1]

zz=xt[2]
fig = plt.figure(9)

ax = fig.gca(projection='3d')

ax.plot(xx,yy,zz,linewidth=1,label='numerical solution')

ax.legend()

plt.show()