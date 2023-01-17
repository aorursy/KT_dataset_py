# import standard python libraries for plotting and calculations

import matplotlib.pyplot as plt

import numpy as np
mylist = []               # creates an empty list 



a = 1 

while a < 11:             # as for the for loop all commands of the while loop are indented

    mylist.append(a)      # the .append attaches 'a' to the end of mylist, so if i started with 

                          # 'mylist=[d,c,b]', it'll now be'mylist = [d,c,b,a]'. We could easily 

                          # do the same thing with 'for a in range(1,11)' instead of 'while'.   

    a = a + 1



print(mylist)             # this command is not part of the while loop so is only executed after the loops finishes
print(mylist[0])  # prints the first element of mylist.
mylist = []



mylist = list(range(2,22,2)) # range(a,b,s) gives all whole numbers 'a + s*n < b' for n=0,1,2...



print(mylist)
mylist = []



mylist = [3*a for a in range(1,11)] # This is called a 'list comprehension', kind of like a for-loop + list creator in one



print(mylist)
mylist = np.zeros(10)

print(mylist)
mylist = np.ones(5)

print(mylist)
myarray = np.array(range(200))  # creates an array with 200 elements from 0 to 199

myarray = myarray/200*4*np.pi   # changes the range to go from 0 to 4pi

myarray2 = np.sin(myarray)      # stores the results of sin(x) in myarray2

#print(myarray2)
myarray = np.array(range(200))             # creates an array with 200 elements from 0 to 199

myarray = myarray/200 *4*np.pi             # changes the range to go from 0 to 4pi

myarray2 = np.sin(myarray)                 # creates a new array with the sine values 

plt.plot(myarray,myarray2,color='red')     # plotting each of the x values against each of the y values, and 

                                           # adjusting the colour of the line



plt.xlabel('My x axis label')              # This labels the x axis

plt.ylabel('My y axis label')

plt.title('My graph\'s title')             # Title for the graph

plt.grid(True)                             # This enables that the grid is displayed

plt.show()
myarray3 = np.cos(myarray)



plt.plot(myarray,myarray2,color='red',label='sin(x)') 

plt.plot(myarray,myarray3,color='blue',label='cos(x)')

plt.plot(myarray,np.sin(myarray)**2,linestyle='dashed',color='green',label='sin(x)**2')

plt.legend()

plt.show()
plt.plot(myarray,myarray2,color='red',label='sin(x)') 

plt.plot(myarray,myarray2,'o',color='blue',label='actual points') # display the data points

plt.axis([0,2,-1.05,1.05])  # change the axis to only show part of the graph.

plt.legend()

plt.show()
tvals = np.arange(0,21,1)              # creates an array with 20 elements from 0 to 20

yvals = np.exp(-tvals/5)               # creates a new array with the y values 

plt.plot(tvals,yvals,color='blue')     # plotting each of the x values against each of the y values, and 

                                       # adjusting the colour of the line



plt.xlabel('t')                        # This labels the x axis

plt.ylabel('exp(-t/5)')                # This labels the y axis

plt.title('exp(-t/5) over t')          # Title for the graph

plt.grid(True)                         # This enables that the grid is displayed

plt.show()
# Distance travelled by non-relativistic muons 

# Calculate the distance and comment on your result here. 



# Calculation



c = 3.00e8              # Speed of light

u = 0.98*c              # Speed of muons

t0 = 2.197e-6           # mean lifetime of particle



x = u*t0                # Insert your code here.



print('The muon travels', x , 'metres.')

print('')

print('Our comment on the implication of this result:')
# Calculate the number of muons reaching sea level on Earth and print your comment here.



# (a)



# Set initial values:

c = 3.00e8                # Speed of light

u = 0.98*c                # Speed of muons

t0 = 2.197e-6             # Mean lifetime of particle

N0 = 5.10e20              # Number of muons created per minute

t = 15000/u               # Calculate the time to reach sea level, time = distance/speed.



N = N0*np.exp(-t/t0)      # Calculate the number of muons that reach the earth per minute



print('The calculated number of muons which reach the earth per minute is:', N)



# (b)



A = 5.101e14              # Surface area of the eath (taken from https://en.wikipedia.org/wiki/Earth)

f = 10**4                 # flux of muons that arrive at sea level per square metre per minute on average



Nprime = A*f              # Total number of muons that reach the earth, Surface Area x Flux



print('The real number of muons which reach the earth per minute is:', Nprime)
# part (a) and (b): Calculate 'velocity' and 'gamma' arrays here. 



c = 3.00e8            

velocity = np.linspace(0,c,10000,endpoint=False)

gamma = 1/(np.sqrt(1-(velocity**2/c**2)))



# part (c) Plot the relationship between velocity and the Lorentz factor.



plt.plot(velocity,gamma,color='blue')             # plotting each of the x values against each of the y values, and 

                                                  # adjusting the colour of the line



plt.xlabel('velocity (m/s)')                      # This labels the x axis

plt.ylabel('gamma')                               # This labels the y axis

plt.title('Lorenz factor over velocity')          # Title for the graph

plt.xlim(0,c)                                     # This sets the range of velocity values shown in the plot.

plt.ylim(0,10)                                    # This sets the range of gamma values shown in the plot.

plt.grid(True)                                    # This enables that the grid is displayed

plt.show()



# part (d) Print the gamma value associated with v = 0.98c.



gamma1 = 1/(np.sqrt(1-((0.98*c)**2/c**2)))

print('The value of gamma when v = 0.98c is:', gamma1)
# Calculate the number of muons here and print your result.



# Set initial values:

c = 3.00e8                               # Speed of light

u = 0.98*c                               # Speed of muons

t0 = 2.197e-6                            # Mean lifetime of particle

N0 = 5.10e20                             # Number of muons created per minute

gamma = 1/(np.sqrt(1-(u**2/c**2)))       # Lorenz factor

t = 15000/u                              # Observed time to reach sea level, time = distance/speed

tau = t*(np.sqrt(1-(0.98**2)))           # Proper time



N = N0*np.exp(-tau/t0)                   # Calculate the number of muons that reach the earth per minute



print('According to an observer at sea level, and accounting for relativity.')

print('The calculated number of muons which reach the earth per minute is:', N)
# Calculate the number of muons here, print the calculated result and the short discussion. 



# Set initial values:

c = 3.00e8                               # Speed of light

u = 0.98*c                               # Speed of muons

t0 = 2.197e-6                            # Mean lifetime of particle

N0 = 5.10e20                             # Number of muons created per minute

gamma = 1/(np.sqrt(1-(u**2/c**2)))       # Lorenz factor

L = 15000/gamma                          # Contracted length

t = L/u                                  # Calculated time to reach sea level, time = distance/speed.



N = N0*np.exp(-t/t0)                     # Calculate the number of muons that reach the earth per minute



print('According to the muons, and accounting for relativity.')

print('The calculated number of muons which reach the earth per minute is:', N)
# Insert your calculation, plot, comment here. [25 points]



# Set initial values:

c = 3.00e8                                         # Speed of light

u = 0.98*c                                         # Speed of muons

t0 = 2.197e-6                                      # Mean lifetime of particle

N0 = 5.10e20                                       # Number of muons created per minute



# 1.



KE_relative = 2.00e9                               # Kinetic Energy of the muon

E0 = 0.10e9                                        # Rest Mass Energy of the muon



dilationfactor = (KE_relative/E0) + 1

print('The time dilation factor of a muon with a kinetic energy of 2GeV is:', dilationfactor)



# 2.



KE = np.linspace(1,20e9,10000,endpoint=False)      # Generating an array for Kinetic Energy

dilationfactor = (KE/E0) + 1                       # Generating an array for dilation factor.

v = c*np.sqrt(1-(1/dilationfactor**2))             # Rearranged formula to find velocity of the muon

t0_prime = dilationfactor*t0                       # The observed mean lifetime of particle

t = 15000/v                                        # Observed time to reach sea level, time = distance/speed.



survivalprob = np.exp(-t/t0_prime)                 # Formula for the probability of a muon reaching sea level: N/N0



plt.plot(KE,survivalprob,color='blue')             # Plotting the kinetic energy versus the survival probability



plt.xlabel('Kinetic Energy')                       # This labels the x axis

plt.ylabel('Survival Rate')                        # This labels the y axis

plt.title('Survival Rate over Kinetic Energy')     # Title for the graph

plt.grid(True)                                     # This enables that the grid is displayed

plt.show()

 

# 3.



restmass = 0.1                                     # Rest Mass Energy of the muon in GeV

prob = 0.9                                         # 90% chance for muon to reach sea level

gamma = np.sqrt(1+(15000/(c*t0*np.log(prob)))**2)  # Rearranged formula for the Lorenz Factor(gamma), from v and t

E = gamma*restmass                                 # Formula for total energy = rest mass energy + kinetic energy

print('Minimum total energy required to generate a muon with a 90% survival rate is:', E, 'GeV')