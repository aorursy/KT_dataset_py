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
import matplotlib.pyplot as plt

import numpy as np

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
# Write your code here to create arrays and make the plot. 

t = np.linspace(0,20,100)

y = np.exp(-t/5)

plt.xlabel('time (s)')    

plt.ylabel('exp(-t/5) (s)')

plt.title('exp(-t/5) VS time (0 - 20)')  

plt.grid(True)   

plt.plot(t,y) 

plt.show()
# Distance travelled by non-relativistic muons 

# Calculate the distance and comment on your result here. 

# Insert your code here.

# Calculation



T0 = 0.000002197

c = 3*10**8

v = 0.98*c 

x = v*T0

print(x)



# Print your comment here.



print('The muon travels', x , 'metres.')

print('')

print('My comment on the implication of this result: Because theyve only travelled 645.9 m we wouldnt actually be able to detect the majority of the muons, only the ones that can make the 15000 m journey to the surface can be examined and recorded.')

# Calculate the number of muons reaching sea level on Earth and print your comment here.



# (a)

NO = 5.1*10**20

T0 = 0.000002197

c = 3*10**8

v = 0.98*c 

d = 15000

t = 15000/v

print(t)





N = NO*np.exp((-t/T0))

print(N)

print('The total number of muons that make it to the surface every minute is', N ,)

print('')





# (b)

r = 6371000

AE = 510.1*10**6*(1000**2)

N1 = 10**4*AE

print(N1)



print('My comment on the implication of this result: The results are very different because we didnt consider the relitivistic conditions with this calculation, in order for a more accurate result you need to consider the relitivistic conditions on the difference between the muons perspective and our perspectives on earths surface')                                                                             
# part (a) and (b): Calculate 'velocity' and 'gamma' arrays here. 



c = 3*10**8           

velocity = np.arange(0,(3*10**8), 3999986.667)    

gamma =  1/(np.sqrt(1-(velocity**2)/(c**2))) 

print(velocity)

print(gamma)       





# part (c) Plot the relationship between velocity and the Lorentz factor.

plt.xlabel('Velocity (m/s)')    

plt.ylabel('Lorentz Factor (gamma)')

plt.title('velocity VS Lorentz factor')  

plt.xlim(0,c)    

plt.ylim(0,10) 

plt.grid(True)   

plt.plot(velocity,gamma) 

plt.show()



# part (d) Print the gamma value associated with v = 0.98c.

v1 = 0.98*c

Γ =  1/(np.sqrt(1-(v1**2)/(c**2))) 

print(Γ)

print('The gamma value associated with v = 0.98c is', Γ ,'Which can be seen on the graph as the point where our velocity begins to increase exponentially more toward infinity.')

# Calculate the number of muons here and print your result. 

t1 = t/Γ

print(t1)

N1 = NO*np.exp((-t1/T0))

print(N1)



print('There will be', N1 ,'muons that reach the earths surface from the perspective of an observer at sea level.')
# Calculate the number of muons here, print the calculated result and the short discussion. 

L0 = d

L = L0/Γ

print(L)



#Time of their travel = 

time=L/v

print(time)



#Number of Muons that hit the ground

N2 = NO*np.exp((-time/T0))

print(N2)

print()



print('From the muons perspective there will be', N2 ,'that reach the Earths surface.')

print('There are the same amount of muons hitting the ground from the muons perspective as the sea-level obeservers perspective because both of these take into consideration the time dilation factor (Gamma) which means even though the muons perspective is length contraction and the sea-level observers is time dilation they will be changed by the same factor of gamma. Neither observer sees both length and time contraction, only one or the other.')

print('Compared to the classical case there will be many many more muons reach the earths surface because the length is effected by gamma, so even though they share the same time the length is contracted for the reletivistic muon perspective.')
#1.

E0 = 0.1 #GeV

m = E0/c**2

KE = 2 #GeV

gamma2 = (KE+m*c**2)/(m*c**2)

print(gamma2)

print('The time dilation factor of a muon with kinetic energy of 2 GeV is', gamma2)



#2.

#M is 1 because it's 1 muon

M = 1

mass = 0.1

Kinetic = np.linspace(0.0000000000000001,50,100)

gamma3 = (Kinetic/mass)+1

vmuon = c*(np.sqrt(1-(1/gamma3)))

tmuon = 15000/vmuon

#Muon hitting ground at different kinetic energies

Number = M*np.exp(-tmuon/(T0*gamma3))



plt.xlabel('Kinetic Energy of Muon (GeV)')    

plt.ylabel('Probability of reaching ground')

plt.title('Kinetic energy of Muon vs probability of that muon reaching the ground') 

plt.xlim(0,50)

plt.ylim(0,1) 

plt.grid(True)   

plt.plot(Kinetic,Number) 
#Zoomed in graph for part c

plt.xlabel('Kinetic Energy of Muon (GeV)') 

plt.ylabel('Probability of reaching ground')

plt.title('Kinetic energy of Muon vs probability of that muon reaching the ground') 

plt.xlim(0,20)

plt.ylim(0,1) 

plt.grid(True)   

plt.plot(Kinetic,Number) 
#3.

print('From looking at the graph I have produced I can approximate that the the minimum total energy required for a muon to have where they 90% chance to reach the Earths surface must be around 20 GeV per muon, this means that a cosmic ray with 5.1×10**20 muons in it must have about 1.02*10**22 GeV in total energy for each muon to have a 90% chance of reaching the Sea-level of the Earth.')  