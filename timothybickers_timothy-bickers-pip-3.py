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
#print(myarray2)
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
timearray = np.linspace(0,20,200)   # Time array from 0 to 20 with 200 steps
exparray = np.exp(-timearray/5)     # Array of the exponential as a function of -t/5

plt.plot(timearray,exparray)
plt.xlabel('Time')    # This labels the x axis
plt.ylabel('exp(-t/5)') #This labels the y axis
plt.title('Graph of function exp(-t/5)')   # Title for the graph
plt.grid(True)                   # This enables that the grid is displayed
# Distance travelled by non-relativistic muons 
# Calculate the distance and comment on your result here. 

# Calculation
c = 3 * 10**8                   #Speed of light  
velocitymuon = 0.98 * c         #Velocity of a muon
meanlifetime = 2.197 * 10**-6   #Mean lifetime of a muon
x = velocitymuon * meanlifetime  #Distance travelled by non-relatavistic muons.

print('The muon travels', x , 'metres.')
print('')
print('Our comment on the implication of this result: This implies that for these non-relavistic muons generated in Earths upper atmosphere the average muon wouldnt survive long enough to make the trip to sea level')
# Calculate the number of muons reaching sea level on Earth and print your comment here.

# (a) 
muonscreated = 5.1 * 10**20    #Muons created in the atmosphere per minute at 15 km  above sea level
c = 3 * 10**8                  #Speed of light
velocitymuon = 0.98 * c        #Assumed muon speed
meanlifetime = 2.197 * 10**-6  #Mean lifetime of a muon
timetoearth = 15000/ velocitymuon #Time taken to reach sea level from 15 km for muon

muonsremaining = muonscreated * np.exp(-timetoearth/meanlifetime)  #Number of muons that make it to sea level
print('(a)')
print('Muons that reach sea level per minute:',muonsremaining)
print('Proportion of Muons that reach sea level:',muonsremaining/muonscreated)


# (b) 
muonflux = 10**4               #Flux of muons at sea level
earthradius = 6371000          #Approximate radius of the earth
earthsurfacearea = 4 * np.pi * earthradius**2    #Surface area of the earth
totalmuons = muonflux * earthsurfacearea         #Total number of muons that actually reach the surface of the earth
print('(b)')
print('Muons that actually reach sea level per minute:',totalmuons)
print('Proportion of Muons that actually reach sea level:',totalmuons/muonscreated)
print()
print('This means that in truth, a far greater number and proprtion of muons reach the surface of the earth when accounting for relavistic effects. This means that relavistic effects play a significant role.')
# part (a) and (b): Calculate 'velocity' and 'gamma' arrays here. 

c =  3 * 10**8                                 # Speed of light
velocity = np.linspace(0,c,50,False)           # Velocity array from 0 to c in steps of 1 ms^-1
gamma =  1/((1-(velocity**2/c**2)))**(1/2)     # Lorentz factors array


# part (c) Plot the relationship between velocity and the Lorentz factor.

plt.plot(velocity,gamma)
plt.xlabel('Velocity')    # This labels the x axis
plt.ylabel('gamma') #This labels the y axis
plt.title('Graph of velocity vs gamma')   # Title for the graph
plt.grid(True)                   # This enables that the grid is displayed


# You can set the ranges you want to plot with the following two commands:
#plt.xlim(0,c)    #This sets the range of velocity values shown in the plot.
#plt.ylim(0,10)   #This sets the range of gamma values shown in the plot.



# part (d) Print the gamma value associated with v = 0.98c.

gammamuon =  1/((1-((0.98*c)**2/c**2)))**(1/2)
print('Gamma of Muon at velocity 0.98c:', gammamuon)


# Calculate the number of muons here and print your result. 

observedmeanlifetime = meanlifetime*gammamuon

muonsremaining = muonscreated * np.exp(-timetoearth/observedmeanlifetime)  #Number of muons that make it to sea level
print('(a)')
print('Muons that reach sea level per minute:',muonsremaining)
print('Proportion of Muons that reach sea level:',muonsremaining/muonscreated)

# Calculate the number of muons here, print the calculated result and the short discussion. 

contractedlength = 15000/gammamuon
timetoearth = contractedlength/velocitymuon

muonsremaining = muonscreated * np.exp(-timetoearth/meanlifetime)  #Number of muons that make it to sea level
print('(a)')
print('Muons that reach sea level per minute:',muonsremaining)
print('Proprtion of Muons that reach sea level:',muonsremaining/muonscreated)

print('(b)')
print('The number of muons that survive are the same when in the reference frame of earth or the muon. This is because when in the reference frame for earth the observed mean life time is increased by a factor of gamma of muon, which decreases t/T0 by a factor of gamma, for reference frame of muon this decreases the observed length by a factor of gamma, which decreases time to earth by factor of gamma, which decreases t/T0 by a factor of gamma. These changes in observed mean life time and time to earth are proprtional and thus surviving muons are equal.These two cases are much larger than the classical case with about 10^8 time more muons surviving per minute. ') 

restmass = 0.1
KE = 2
gamma1 = (KE/restmass) + 1
print('(a)')
print(gamma1)
KE = np.arange(0.0000001,120,0.1)    #Kinetic energy array

gamma = (KE/restmass) + 1               #gamma equation
velocity = c*(1-((1/gamma)**2))**0.5    #Relavtistic velocity
contractedlength = 15000/gamma          #Relavistic length 
timetoearth = contractedlength/velocity #Time for muon to reach earth

probability = np.exp(-timetoearth/meanlifetime)  #Probability of survival array

plt.plot(KE,probability)
plt.xlabel('Kinetic Energy of Muon')    # This labels the x axis
plt.ylabel('Probability of Survival') #This labels the y axis
plt.title('Graph of Kinetic Energy vs Survival Probability')   # Title for the graph
plt.grid(True)                   # This enables that the grid is displayed
plt.xlim(0,)    
plt.ylim(0,1)   #This sets the range of probability values shown in the plot.

print('(b)')




gamma2 = ((((-15000/(c*meanlifetime*np.log(0.9)))**2) +1))**0.5   #Rearrangment of known equations to find gamma from constants and probability
minimuntotalenergy = gamma2*restmass                #To find the minimun energy value, note that this is different to kinetic enrgy equation
print('(c)')
print('Minimun total energy for a cosmic event to have a 90% chance of reaching sea level')
print(minimuntotalenergy)

