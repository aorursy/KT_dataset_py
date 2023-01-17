# import standard python libraries for plotting and calculations

import matplotlib.pyplot as plt

import numpy as np
mylist = []          # creates an empty list



a = 1 

while a < 11:        # as for the for loop all commands of the while loop are indented

    mylist.append(a) # the .append attaches 'a' to the end of mylist, so if i started with 

                     #'mylist=[d,c,b]', it'll now be'mylist = [d,c,b,a]'. We could easily 

                     # do the same thing with 'for a in range(1,11)' instead of 'while'.   

    a = a + 1



print(mylist)        # this command is not part of the while loop so is only executed after the loops finishes
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

print(myarray2)
x = np.arange(0,10,0.1)



print(x)



x = np.linspace(0,10,100)



print(x)
myarray = np.array(range(200))   # creates an array with 200 elements from 0 to 199

myarray = myarray/200 *4*np.pi   # changes the range to go from 0 to 4pi

myarray2 = np.sin(myarray)       # creates a new array with the sine values 

plt.plot(myarray,myarray2,color='red') # plotting each of the x values against each of the y values, and 

                                       # adjusting the colour of the line



plt.xlabel('My x axis label')    # This labels the x axis

plt.ylabel('My y axis label')

plt.title('My graph\'s title')   # Title for the graph

plt.grid(True)                   # This enables that the grid is displayed

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
t = np.linspace(0,20,200) # creates an array of 200 elements evenly distributed bewteen 0 and 20



y = np.exp(-t/5) # defines the exponential function



plt.plot(t,y) # plots the exponential function



plt.title("Plot of exp(-t/5) between t = 0 and t = 20") # Titles the graph



plt.xlabel('t') # Labels the x axis



plt.ylabel('exp(-t/5)') # Labels the y axis



plt.grid() # Creates gridlines



plt.show() 

t0 = 2.197e-6 # mean lifetime of muon in s

c = 3e8       # speed of light in m/s

v = 0.98*c    # speed of particle in m/s



x = t0*v



print('The muon travels', x , 'metres.')

# Calculate the number of muons reaching sea level on Earth and print your comment here.



# (a)



t0 = 2.197e-6 # mean lifetime of muon in s

c = 3e8       # speed of light in m/s

v = 0.98*c    # speed of particle in m/s

N0 = 5.1e20   # number of muons created per minute



t = 15000/v   # calculates the time taken to reach earth



N = np.format_float_scientific(N0*np.exp(-t/t0), precision=3) # calculates the number of muons reaching the surface per minute 

                                                              # puts this number into scientific notation to three decimal placs



print(N, 'muons would reach sea level per minute without accounting for relativistic effects.')



                                                                                
c = 3e8 # speed of light in m/s



velocity = np.arange(0,c,c/200) # creates an array of 200 values from 0 up to but not including the speed of light

                                # the interval between each value is c/200



gamma = 1/np.sqrt(1-(velocity**2/c**2)) # calculates the value of gamma depending on the velocity



plt.plot(velocity,gamma) # plots the value of gamma against the velocity

plt.title("Value of Lorentz factor as it depends on velocity") # titles the graph

plt.xlabel('velocity (m/s)') # labels the x axis

plt.ylabel('Lorentz factor') # labels the y axis

plt.xlim(0,c)   # sets the range of velocity values shown in the plot

plt.grid() # creates gridlines

plt.show() 



gamma = 1/np.sqrt(1-((0.98*c)**2/c**2)) # reassigning gamma variable to calculate the Lorentz factor for 0.98c 



print("At 0.98c, the Lorentz factor is", gamma) # prints this value



t0 = 2.197e-6 # mean lifetime of muon in s

c = 3e8       # speed of light in m/s

v = 0.98*c    # speed of particle in m/s

N0 = 5.1e20   # number of muons created per minute



gamma = 1/np.sqrt(1-(v**2/c**2)) # calculates the value of gamma depending on the velocity



t = 15000/v # time as calculated before



tr = t/gamma # time accounting for the effects of time dilation 



N = N0*np.exp(-tr/t0) # calculates number of muons that reach the surface



print(N, 'muons would reach sea level per minute, this time accounting for relativistic effects.')

t0 = 2.197e-6 # mean lifetime of muon in s

c = 3e8       # speed of light in m/s

v = 0.98*c    # speed of particle in m/s

N0 = 5.1e20   # number of muons created per minute



gamma = 1/np.sqrt(1-(v**2/c**2)) # calculates the value of gamma depending on the velocity



d = 15000/gamma # contracted distance when relativity is taken into account



t = d/v #time taken to travel contracted distance



N = N0*np.exp(-t/t0) # calculates number of muons that reach the surface



print("From the perspective of the muons", N, "muons would reach sea level per minute")

# a



c = 3e8               # speed of light in m/s

KE = 2                # kinetic energy in GeV

restmass = 0.1        # rest mass in GeV

t0 = 2.197*(10**-6)   # half life of muon in s



gamma = (KE/restmass) + 1 #calculates the time dilation factor for a muon with kinetic energy 2GeV



print("a.", gamma)



# b 



KE = np.arange (0.0001,150,0.01) #creates an array of values from 0.0001 (close to 0) up to 150, with intervals of 0.01



gamma = KE/restmass + 1 #calculates gamma based on kinetic energy and rest mass



v = c*np.sqrt(1-1/gamma**2) #calculates speed based on gamma



d = 15000/gamma # contracted distance when relativity is taken into account



t = d/v # time taken to travel contracted distance



# the probability of survival is the proportion of muons that actually make it to earth versus the amount created 

# from the equation given, this is equal to N/N0 =  (N0*np.exp(-t/t0))/N0 = np.exp(-t/t0)



survival = np.exp(-t/t0) #calculates probability of survival



plt.plot(KE,survival) #plots survival probability against kinetic energy

plt.title("b. Survival probability of a muon depending on kinetic energy\n") # titles the graph

plt.xlabel('Kinetic energy (GeV)') # labels the x axis

plt.ylabel('Survival probability') # labels the y axis

plt.grid() # creates gridlines

plt.show() 



# c



# combining these equations:

# survival probability,s = np.exp(-t/t0)

# velocity, v = c*np.sqrt(1-1/gamma**2)

# time dilation factor, gamma = KE/restmass + 1

# we get:



survival = 0.9 #defines the survival probability as 90%



gamma = np.sqrt((15000/(c*np.log(survival)*t0))**2+1)



print("c. \nThe new time dilation factor is", gamma)



# Total energy is gamma * restmass, so:



minimum_E = gamma * restmass #calculates minimum total energy



print("The minimum energy needed for the muon to have a 90% chance of surviving is", minimum_E,"GeV")

print("This is consistent with my graph.")
