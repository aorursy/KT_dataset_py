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

t_array = np.linspace(0,20) #array of values between 0 and 20
exp = np.exp(-t_array/5) #exp(-t/5) array of t-values between 0 and 20
plt.plot(t_array,exp,color='red')

plt.xlabel('t')    # This labels the x axis
plt.ylabel('exp(-t/5)')
plt.title('exp(-t/5)/t')   # Title for the graph
plt.grid(True)                   # This enables that the grid is displayed
plt.show()
# Distance travelled by non-relativistic muons 
# Calculate the distance and comment on your result here. 

# Calculation

t0 = 2.197 * 10**(-6)
v = 0.98 * 3 * 10**8
x = t0 * v

# Print your comment here.

print('The muon travels', x , 'metres.')
print('')
print('Our comment on the implication of this result: Muon decays long before reaching sea level')

# Calculate the number of muons reaching sea level on Earth and print your comment here.

# (a)
x = 15000
v = 0.98 * 3 * 10**8
t_total = x / v
t0 = 2.197 * 10**(-6)
n0 = 5.1 * 10**20
n = n0 * np.exp(-t_total/t0)
print(int(n))
#result suggest no muons actually reach sea level

# (b)
#The only possible way for this many muons to be reaching sea level is for the muons to be relativistic
                                                                                
# part (a) and (b): Calculate 'velocity' and 'gamma' arrays here. 

c =            3 * 10**8
velocity =     np.arange(0, c, 1000000)
gamma =        1 / (1 - (velocity/c)**2)**(1/2)


# part (c) Plot the relationship between velocity and the Lorentz factor.
plt.plot(velocity,gamma,color='red')

plt.xlabel('velocity')    # This labels the x axis
plt.ylabel('gamma')
plt.title('lorentz factors')   # Title for the graph
plt.grid(True)                   # This enables that the grid is displayed
plt.show()
# You can set the ranges you want to plot with the following two commands:
# plt.xlim(0,c)    This sets the range of velocity values shown in the plot.
# plt.ylim(0,10)   This sets the range of gamma values shown in the plot.



# part (d) Print the gamma value associated with v = 0.98c.
lorentz = 1 / np.sqrt(1 - ((0.98*c)/c)**2)
print(lorentz)

# Calculate the number of muons here and print your result. 

x = 15000
v = 0.98 * 3 * 10**8
t_total = x / v
gamma = 1 / np.sqrt(1 - ((v)/c)**2)
dialated = t_total * gamma
t0 = 2.197 * 10**(-6)
n0 = 5.1 * 10**20
n = n0 * np.exp(-dialated/t0)
print(int(n))


# Calculate the number of muons here, print the calculated result and the short discussion. 
x = 15000
v = 0.98 * 3 * 10**8
gamma = 1 / np.sqrt(1 - ((v)/c)**2)
contracted = x / gamma
t_total = x / v
t0 = 2.197 * 10**(-6)
n0 = 5.1 * 10**20
n = n0 * np.exp(-t_total/t0)
print(int(n))



# Insert your calculation, plot, comment here. [25 points]
eV = 1.6022 * 10**(-19)
# 1.
k_e = 2 * 10**(-6) * eV
rest_mass = 0.1 * 10**(-6) * eV
gamma = ((k_e + rest_mass)/ rest_mass)
print(gamma)

# 2.
prob = np.linspace(0, 1.0, 0.01)
v = np.linspace(0, c, 100)
gamma = 1 / np.sqrt(1 - ((v)/c)**2)
KE = rest_mass * (gamma - 1)
print(prob)
print()
print(KE)
plt.plot(prob,KE,color='red')

plt.xlabel('prob')    # This labels the x axis
plt.ylabel('kE')
plt.title('KE/prob')   # Title for the graph
plt.grid(True)                   # This enables that the grid is displayed
plt.show()
# 3.

