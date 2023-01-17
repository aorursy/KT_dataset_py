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
# Write your code here to create arrays and make the plot. 

timearray = np.array(range(21))   # 
exparray = np.exp(-timearray/5)   #  
plt.plot(timearray,exparray,color='red') # plotting each of the x values against each of the y values, and 
                                       # adjusting the colour of the line

plt.xlabel('time (s)')    # This labels the x axis
plt.ylabel('exp(-t/5)')
plt.title('exp(-t/5)')   # Title for the graph
plt.grid(True)                   # This enables that the grid is displayed
plt.show()
# Distance travelled by non-relativistic muons 
# Calculate the distance and comment on your result here. 

# Calculation
T0 = 2.197*10**(-6)

speed = 0.98*3*10**8

x = T0 * speed  # Insert your code here.

# Print your comment here.

print('The muon travels', x , 'metres.')
print('')
print('Our comment on the implication of this result: very small percentage would reach earth surface')

# Calculate the number of muons reaching sea level on Earth and print your comment here.

# (a)
time = 15000/speed

number= 5.1*10**20

N=number*np.e**(-time/T0)

print(' the number of muons reaching sea level is ',N, '.')


# (b)
#  It meast the result is wrong without accounting for relativistic effects.
#. 41884445975/10^4 = 4.2x10^6 across then entire earch surface, and far far less than 10^4/sqm.                                                                
# part (a) and (b): Calculate 'velocity' and 'gamma' arrays here. 

c = 3*10**8          # Insert your code here.
velocity = np.arange(0,c,c/50)    # Insert your code here.
gamma =  1/(np.sqrt(1-(velocity/c)**2))      # Insert your code here.


# part (c) Plot the relationship between velocity and the Lorentz factor.

plt.plot(velocity,gamma)

# You can set the ranges you want to plot with the following two commands:
plt.xlim(0,c)    #This sets the range of velocity values shown in the plot.
plt.ylim(0,10)   #This sets the range of gamma values shown in the plot.
plt.title('velocity(x) vs lorentz factor(y)')

# part (d) Print the gamma value associated with v = 0.98c.
gamma1 =  1/(np.sqrt(1-(0.98*c/c)**2))

print ('value of y for muon traveling at 0.98c', gamma1)

# Calculate the number of muons here and print your result. 

gamma2 =  1/(np.sqrt(1-((0.98*c)/c)**2))      

time = 15000/speed

number= 5.1*10**20

N=number*np.e**(-time/(T0*gamma2))

print (' the number of muons reaching sea level is ',N, '.')


# Calculate the number of muons here, print the calculated result and the short discussion. 

speed = 0.98*3*10**8

gamma1 =  1/(np.sqrt(1-(0.98)**2))

Legnth = 15000/gamma1

time = Legnth/speed

number= 5.1*10**20

N=number*np.e**(-time/T0)

print(' the number of muons reaching sea level is ',N, '.')


mass=1.88*10**-28

MC2=0.1/(6.242*10**9)

#a)

KE2=2*3.2*10**-10

TF1 = (KE2+ MC2)/MC2

#b)

number= 5.1*10**20

KEI = 2*10**-10

KE = np.arange(0.001,50,0.001)   

KE = KE*3.2*10**-10

TIMEFACTOR = (KE  + MC2)/MC2

v = c*(np.sqrt(1+(1/TIMEFACTOR)**2))

time = 15000/v

hits = number*(np.e**(-time/(T0*TIMEFACTOR)))

survivalprobability = hits/number

# part (c) Plot the relationship between velocity and the Lorentz factor.

plt.plot(KE,survivalprobability)

plt.xlabel('energy in joules')    
plt.ylabel('hit rate')
plt.title('chance a muon with x KE will reach the sea from 15km')   
plt.grid(True)

plt.xlim(0,50*3.2*10**-10)    #This sets the range of velocity values shown in the plot.
plt.ylim(0,1)

print(' the time dialtion factor of a muon with KE=2GeV is ',TF1, '.')


# C) aprox 0.3*10^-8J, based off of the graph


