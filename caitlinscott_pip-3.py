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
print(myarray)
myarray = myarray/200*4*np.pi   # changes the range to go from 0 to 4pi
print(myarray)
myarray2 = np.sin(myarray)      # stores the results of sin(x) in myarray2
#print(myarray2)
x = np.arange(0,10,0.1)
#print(x)
x = np.linspace(0,10,100)
print(x)
myarray = np.array(range(200))   # creates an array with 200 elements from 0 to 199
myarray = myarray/200 *4*np.pi   # changes the range to go from 0 to 4pi
myarray2 = np.sin(myarray)       # creates a new array with the sine values 
plt.plot(myarray,myarray2,color='red') # plotting each of the x values against each of the y values, and 
                                       # adjusting the colour of the line
    #y axis is the second input. 

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
# Write your code here to create arrays and make the plot. 
t = np.linspace(0,20, 100)

expFunction = np.exp(-t/5)
# Plot
plt.plot(t, expFunction, color='cyan',label='exp(-t/5)')

plt.xlabel('t (unit)')    
plt.ylabel('exp(-t/5) (unit)')
plt.title('Plot of exp(-t/5)')  
plt.legend()
plt.grid(True)                   
# Distance travelled by non-relativistic muons 
# Calculate the distance and comment on your result here. 

# Calculation

time =  2.197e-6
c = 300000000
velocity = 0.98*c
x =   velocity*time

# Print your comment here.

print('The muon travels', x , 'metres.')
print('')
print('This implies that the average non relativistic muons created in the upper atmosphere will never reach sea level, as they will attentuate in the atmosphere. ')

distanceToSealLevel = 1.5e4
timeToReachSeaLevel = distanceToSealLevel/velocity
initialNumberOfMuons = 5.1e20
T0 = 2.197e-6

###
numberOfMuonsReachingSeaLevel = initialNumberOfMuons*(np.exp(-timeToReachSeaLevel/T0))
print(numberOfMuonsReachingSeaLevel)
#b.
earthSurfaceArea = 5.10e14
flux = numberOfMuonsReachingSeaLevel / earthSurfaceArea
print("The flux is", flux, "m^-2")
# Factor of Difference between calculated and observed
differenceFactor = 1e4/flux
print("The observed flux is greater than that calculated by a factor of",differenceFactor)
# Calculate the number of muons reaching sea level on Earth and print your comment here.

# (a)


# (b) The flux amount of muons that are found to reach Earth when ignoring the relativistic effects is significantly less 
#than that observed, approximately 120000000 more than what is calculated. This suggests that a relativity correction factor 
#is required to account for this observation. 

                                                                                
# part (a) and (b): Calculate 'velocity' and 'gamma' arrays here. 

c = 3e8     

velocity1 = np.linspace(0, (c-1),100) 
#print(velocity1)

gamma = np.array(range(100)) 

#for i in range(100):  
gamma=(1/(np.sqrt(1-(pow(velocity1, 2))/(pow(c, 2)))))
                           


# part (c) Plot the relationship between velocity and the Lorentz factor.
plt.plot(velocity1, gamma, color='cyan',label='???')

# You can set the ranges you want to plot with the following two commands:
plt.xlim(0,c)    #This sets the range of velocity values shown in the plot.
plt.ylim(0,10)   #This sets the range of gamma values shown in the plot.

plt.xlabel('Velocity (ms^-1)')    # This labels the x axis
plt.ylabel('Gamma')
plt.title('Dependance of Gamma on Velocity')   # Title for the graph
plt.grid(True)                   # This enables that the grid is displayed
plt.show()




# part (d) Print the gamma value associated with v = 0.98c.
 
gammaAt98c =1/(np.sqrt(1-(pow(0.98, 2))))
print(gammaAt98c)

# Calculate the number of muons here and print your result. 

L0 = 1.5e4  
N0 = 5.1e20
T0 = 2.197e-6
cSquared = c**2
timeToReachSeaLevel = L0/velocity
gamma1=(1/(np.sqrt(1-(pow(velocity, 2))/cSquared)))

time_prime = timeToReachSeaLevel/gamma1

T0_prime = T0 *gamma1
###
numberOfMuonsReachingSeaLevel = N0*(np.exp(-(time_prime/T0_prime)))
print("Number of muons reaching sea level considering the relativistic correction factor: ",numberOfMuonsReachingSeaLevel)



# Calculate the number of muons here, print the calculated result and the short discussion. 
#a.
L0 = 1.5e4
L = L0/gamma1
T0 = 2.197e-6
N0 = 5.1e20
RelativetimeToReachSeaLevel = L/velocity
T0_prime = T0 *gamma1 #(The relative time to reach sea level must also be considered relativistically)
###
numberOfMuonsReachingSeaLevel = N0*(np.exp(-(RelativetimeToReachSeaLevel/T0_prime)))
print(numberOfMuonsReachingSeaLevel)

#b. The number of muons that reach the ground from their frame of reference, which includes relativistic effects, 
#   is much greater than that calculated without relativistic effects, i.e. it is the same magnitude as that observed.
#   This displays the validity of the calculation in terms of the theory pf relativity. Furthermore, the relativistic
#   correction factor applied by means of time dilation in Earth's reference frame, and length contraction in terms of
#   the reference frame of the muons. 
# Insert your calculation, plot, comment here. [25 points]

# 1.
c= 3e8
d=1.5e4
cSquared=c**2
KE_GeV = 2
E0 = 0.1
time_dilation_factor = (KE_GeV/E0)+1
print("The time dilation factor for a muon with 2GeV of kinetic energy is ", time_dilation_factor)

# 2.

# Range of kinetic energies
KE_GeV_array = np.linspace(0,20,100)
# Range of gammas based on range of EK
gamma_array = (KE_GeV_array/E0)+1
# Velocity from the frame of the muon
muon_v = np.sqrt(cSquared *(1- (pow((1/gamma_array), 2))) )
# Array for time
time = d/muon_v
# Mean time dilated
T0_dilated = gamma_array*T0


# Probabilities with a range of kinetic energies 
probability = np.exp(-time/(T0_dilated))

# Plot
plt.plot(KE_GeV_array, probability)
plt.xlabel('KE (Gev)')    # This labels the x axis
plt.ylabel('Probability')
plt.title('Probability of muons hitting the ground with different KE')   # Title for the graph
plt.grid(True)                   # This enables that the grid is displayed
plt.show()


# 3. do the calculation backward
#0.9=np.exp(-time/(gamma_array*T0)) #rearrange this for time. 
lnprob_T0squared = ((np.log(0.9))*T0)**2
# Velocity required for 90% probability of hitting Earth
v = np.sqrt((d**2)/(lnprob_T0squared + ((d**2)/cSquared)))
print(v)
# Relativistic factor for EK from veloctiy
gamma2=(1/(np.sqrt(1-(pow(v, 2))/cSquared)))
# TOTAL ENERGY
E = (gamma2)*E0
print("The Total energy required for a muon to have a 90% probability of hitting sea level is", E, "GeV")
Print("This matches what is displayed on the graph")

