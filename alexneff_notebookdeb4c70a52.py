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



myarray5 = np.array(range(20))   # creates an array with 20 elements from 0 to 19

myarray6 = (-(myarray5)/5)

myarray6 = np.e**(myarray6) 

plt.plot(myarray5,myarray6,color='red') # plotting each of the x values against each of the y values, and 

                                       # adjusting the colour of the line



plt.xlabel('Time')    # This labels the x axis

plt.ylabel('Y value')    # This labels the y axis

plt.title('Exponential function with respect to time')   # Title for the graph

plt.grid(True) # This enables that the grid is displayed

plt.axis([0,20,0,1])

plt.legend()

plt.show()
# Distance travelled by non-relativistic muons 

# Calculate the distance and comment on your result here. 



# Calculation



c = 3*10**8

v = 0.98*c

t = 2.197*10**-6

x = v*t

# Print your comment here.



print('The muon travels', x , 'metres.')

print('')

print('Our comment on the implication of this result: Without relativistic ideas incorporated into this example,this tells us the muons only fall 645 metres, rather than 15 km.')
# Calculate the number of muons reaching sea level on Earth and print your comment here.



# (a)

N_o = 5.1*10**20

T_o = 1.523*10**-6

t = 15000/(0.98*c)

N = N_o*np.e**(-t/(T_o))

print(N)







# (b) It has stated that 10**4 muons / square meter make it down to sea level per minute. This multiplied by the surface of the earth is going to be much much bigger

# than the value we have calculated below (10**6). this is because we neglect the relativistic properties of the muons in travel.
# part (a) and (b): Calculate 'velocity' and 'gamma' arrays here. 



c = 3*10**8    # Speed of light in m/s.

velocity = np.arange(0,c,10**6)    # array for velocity from 0 to c, but not including c

gamma = 1/(np.sqrt(1-np.square(velocity/c)))       # gamma (lorentz factor).





# part (c) Plot the relationship between velocity and the Lorentz factor.



plt.plot(velocity,gamma,color='red')

plt.title('Velocity with respect to Gamma')

plt.xlim(0,c)

plt.ylim(0,15)

plt.legend()

plt.show()



# You can set the ranges you want to plot with the following two commands:

# plt.xlim(0,c)    This sets the range of velocity values shown in the plot.

# plt.ylim(0,10)   This sets the range of gamma values shown in the plot.







# part (d) Print the gamma value associated with v = 0.98c.

print("The value of gamma for a velocity of 0.98c is:")

print("gamma = 1/(sqrt(1-(0.98)^2))")

print("gamma = ",1/np.sqrt(1-np.square(0.98)))
# Calculate the number of muons here and print your result.



N_o = 5.1*10**20

T_o = 1.523*10**-6

t = 15000/(0.98*c)

gamma = 1/(np.sqrt(1-np.square(0.98)))

real_t = t/gamma



N = N_o*np.e**(-real_t/(T_o))

print(N)

# Calculate the number of muons here, print the calculated result and the short discussion.





L_o = 15000

L = L_o/gamma

t2 = L/(0.98*c)

real_t = t2/gamma



N = N_o*np.e**(-real_t/(T_o))

print(N)



 

# the results gathered from the relitivistic calculations are much greather than that of the calculations done in task 4. 

# 
# Insert your calculation, plot, comment here. [25 points]



# 1.



# 2.



# 3.