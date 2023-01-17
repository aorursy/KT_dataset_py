import numpy as np

%matplotlib notebook

import matplotlib.pyplot as plt
# ------------ INITIALISATION ---------------------



# --- INPUT ---



q_m_ratio = 1       # charge-to-mass ratio

b0 = 1             # magnetic field (homogeneous, in z direction) 



v = 10              # enter the correct initial velocity.



phi = 0             # enter the correct phase

rxc = 0            # enter the correct x-coordinate of the centre of the circle

ryc = 0            # enter the correct y-coordinate of the centre of the circle



start_time = 0      # enter the start time of the propagation

N_loops = 2         # enter number of loops the particle should complete 

steps_rev = 1000    # enter the number of time steps per revolution 



# --- calculate values from input variables



omega = q_m_ratio*b0            # calculate the cyclotron frequency (from the input variables)



period_osc = (2*np.pi)/omega    # calculate the period of the oscillation (from the input variables)



end_time = N_loops*period_osc   # calculate the end time of the propagation (from the input variables)



time_steps = N_loops*steps_rev  # calculate the number of time steps (from the input variables)



# --- definition of time array for propagation 



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

# Output: yout - propagated vector of position and velocity



def euler(yin,omega,step):                 #defines the function

    yout = yin + step*der(yin,omega)       #performs the calculation y(out)=y(in)+h𝐹(y(in) as given above .

    return yout                            #returns the new array with calculated values

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
#imports libraries

import numpy as np

%matplotlib notebook

import matplotlib.pyplot as plt



# ----Defining a function called 'plot' for all of the code----

# Input: N_loop - Number of loops the particle will do

#        steps_rev - Number of time steps per revolution

#        v - Initial velocity 

# I'll be able to call this function to plot a new y-t and x-y graph for a given input

def plot (N_loops,steps_rev,v):



    #Defining input variables that I won't be changing later on

    q_m_ratio = 1      # charge-to-mass ratio

    b0 = 1             # magnetic field (homogeneous, in z direction)

    phi = 0            # phase

    rxc = 0            # x-coordinate of the centre of the circle

    ryc = 0            # y-coordinate of the centre of the circle

    start_time = 0     # start time of the propagation



    # --- calculate values from input variables

    omega = q_m_ratio*b0            # cyclotron frequency (from the input variables)

    period_osc = (2*np.pi)/omega    # period of the oscillation (from the input variables)

    end_time = N_loops*period_osc   # end time of the propagation (from the input variables)

    time_steps = N_loops*steps_rev  # number of time steps (from the input variables)



    # --- definition of time array for propagation 

    step = (end_time-start_time)/(time_steps-1)

    time = np.linspace(start_time,end_time,time_steps)



    #ANALYTICAL SOLUTION of the differential equations

    exact_position_x = v/omega*np.sin(omega*time - phi) + rxc   # multiplication with numpy array time

    exact_position_y = v/omega*np.cos(omega*time - phi) + ryc   # yields a numpy array

    exact_velocity_x = v*np.cos(omega*time - phi)

    exact_velocity_y = -v*np.sin(omega*time - phi)



    # initialisation of position and velocity arrays, setting everything to zero

    position_x = np.zeros(time_steps)

    position_y = np.zeros(time_steps)

    velocity_x = np.zeros(time_steps)

    velocity_y = np.zeros(time_steps)



    # --- Definition of the derivative of y(t), F(y(t)) ---

    # Input:  yin :  vector y (numpy array)

    #         omega: cyclotron frequency

    # Output: derivative of vector yin, F(yin) (numpy array)

    def der(yin, omega):                                              # no explicit time dependence

        return np.array([yin[2],yin[3],omega*yin[3],-omega*yin[2]])





    # --- Implementation of Euler method

    # Input: yin - initial vector of position and velocity

    #        omega - cylcotron frequency

    #        step - time step

    # Output: yout - propagated vector of position and velocity

    def euler(yin,omega,step):

        yout = yin + step*der(yin,omega)

        return yout



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

        yin = euler(yin,omega,step)                    # calculation of yout (which immediately overwrites the old yin)

    

        position_x[ii] = yin[0]                        # save the new position and velocity components

        position_y[ii] = yin[1]

        velocity_x[ii] = yin[2]

        velocity_y[ii] = yin[3]



    # x-y plot of cyclotron motion 

    # comparison analytical vs. numerical solution

    

    plt.figure()

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

    

    plt.figure()

    plt.plot(time,position_y,'x',label='numerical position')

    plt.plot(time,exact_position_y,linewidth=1,label='Analytical position')

    plt.axis([0,N_loops*2*np.pi/omega,-1*(position_y[0]+0.1),position_y[0]+0.1])

    plt.title(f"Comparison analytical vs numerical result of y(t)")

    plt.xlabel(f"time [s]") 

    plt.ylabel(f"y position [m]") 

    plt.legend

    plt.show()



#Setting the input variables to what they were originally defined as in previous tasks to check that the output is correct

print("Number of loops: 2\nTime steps per revolution: 1000\nInitial velocity: 10")

plot(2,1000,10)



#Changing the number of loops to 8

print("Number of loops: 8\nTime steps per revolution: 1000\nInitial velocity: 10")

plot(8,1000,10)



#Increasing the number of time steps to 10000 to decrease error in numerical method

print("Number of loops: 8\nTime steps per revolution: 10000\nInitial velocity: 10")

plot(8,10000,10)



#Doubling the initial velocity to 20m/s

print("Number of loops: 8\nTime steps per revolution: 10000\nInitial velocity: 20")

plot(8,10000,20)



#Halving the initial velocity to 5m/s

print("Number of loops: 8\nTime steps per revolution: 10000\nInitial velocity: 5")

plot(8,10000,5)



#imports libraries

import numpy as np

%matplotlib notebook

import matplotlib.pyplot as plt



# ----Defining a function called 'error' for all of the code----

# The only input I will be changing is the number of time steps

# Input: steps_rev - Number of time steps per revolution

def error (steps_rev):



    #Defining input variables that I won't be changing later on

    v = 10             #initial speed

    N_loops = 8        # number of loops

    q_m_ratio = 1      # charge-to-mass ratio

    b0 = 1             # magnetic field (homogeneous, in z direction)

    phi = 0            # phase

    rxc = 0            # x-coordinate of the centre of the circle

    ryc = 0            # y-coordinate of the centre of the circle

    start_time = 0     # start time of the propagation



    # --- calculate values from input variables

    omega = q_m_ratio*b0            # cyclotron frequency

    period_osc = (2*np.pi)/omega    # period of the oscillation 

    end_time = N_loops*period_osc   # end time of the propagation 

    time_steps = N_loops*steps_rev  # number of time steps 



    # --- definition of time array for propagation 

    step = (end_time-start_time)/(time_steps-1)

    time = np.linspace(start_time,end_time,time_steps)



    #ANALYTICAL SOLUTION of the differential equations

    exact_position_x = v/omega*np.sin(omega*time - phi) + rxc   # multiplication with numpy array time

    exact_position_y = v/omega*np.cos(omega*time - phi) + ryc   # yields a numpy array

    exact_velocity_x = v*np.cos(omega*time - phi)

    exact_velocity_y = -v*np.sin(omega*time - phi)



    # initialisation of positio and velocity arrays, setting everything to zero

    position_x = np.zeros(time_steps)

    position_y = np.zeros(time_steps)

    velocity_x = np.zeros(time_steps)

    velocity_y = np.zeros(time_steps)



    # --- Definition of the derivative of y(t), F(y(t)) ---

    # Input:  yin :  vector y (numpy array)

    #         omega: cyclotron frequency

    # Output: derivative of vector yin, F(yin) (numpy array)

    def der(yin, omega):                                              # no explicit time dependence

        return np.array([yin[2],yin[3],omega*yin[3],-omega*yin[2]])





    # --- Implementation of Euler method

    # Input: yin - initial vector of position and velocity

    #        omega - cylcotron frequency

    #        step - time step

    # Output: yout - propagated vector of position and velocity

    def euler(yin,omega,step):

        yout = yin + step*der(yin,omega)

        return yout



    #Input of initial conditions

    position_x[0] = 0.0        # initial position

    position_y[0] = v/omega  

    velocity_x[0] = v          # initial velocity

    velocity_y[0] = 0.0  



    yin = np.zeros(4)          # initialisation of yin



    yin[0] = position_x[0]     # start with initial conditions 

    yin[1] = position_y[0]

    yin[2] = velocity_x[0]

    yin[3] = velocity_y[0]



    #Propagation

    for ii in range(1,time_steps):

        yin = euler(yin,omega,step)                    # calculation of yout (which immediately overwrites the old yin!)

    

        position_x[ii] = yin[0]                        # save the new position and velocity components

        position_y[ii] = yin[1]

        velocity_x[ii] = yin[2]

        velocity_y[ii] = yin[3]

        

    #Calculation of error arrays

    error_x = exact_position_x - position_x

    error_y = exact_position_y - position_y

    error_vx = exact_velocity_x - velocity_x

    error_vy = exact_velocity_y - velocity_y



    #plots the error over time for position

    plt.figure(100)

    plt.plot(time,(error_x**2+error_y**2)**0.5,label= 'Steps per revolution:'+ str(steps_rev))

    plt.title('Error produced through Euler method - position')

    plt.xlabel(f"time [s]") 

    plt.ylabel(f"Error (m)")

    plt.legend()

    plt.show()

    

    #plots the error over time for velocity

    plt.figure(101)

    plt.plot(time, (error_vx**2+error_vy**2)**0.5, label= 'Steps per revolution:'+ str(steps_rev))

    plt.title('Error produced through Euler method - velocity')

    plt.xlabel(f"time [s]") 

    plt.ylabel(f"Error (m/s)") 

    plt.legend()

    plt.show()



#plots error over time for 50, 100, and 1000 time steps per revolution

error(50)

error(100)

error(1000)