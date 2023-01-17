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
x = np.arange(0,10,0.1)
print(x)
x = np.linspace(0,10,100)
print(x)
x = np.arange(1,10,1)
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

t_values = np.linspace(0,20,21)
y_values = np.exp(-1*t_values/5)

plt.plot(t_values,y_values,color='red')
plt.xlabel('t')
plt.ylabel('e^(-t/5)')

# Distance travelled by non-relativistic muons 
# Calculate the distance and comment on your result here. 

# Calculation

#v = d/t

v = 0.98*3*10**8
t = 2.197*10**-6

x = v*t

comment = '''this gives a result in the order of 100m, so the chance that a muon reaches the surfae of 
the earth is extraordinarily small. We know that muons do in fact reach the surface of the earth in a 
much greater abundance, showing that classical relativity cannot explain this phenomenom'''

print('The muon travels', x , 'metres.')
print('')
print(comment)

# Calculate the number of muons reaching sea level on Earth and print your comment here.

# (a)
d = 15000
v = 0.98*3*10**8

No = 5.1*10*20
t = d/v
To = 2.197*10**-6

N = No*np.exp(-t/To)

print('number of muons that reach the surface of the earth:', N)

# (b)
comment = '''this is not nearly enough! We are getting basically 0 muons when we should be getting 
around much more, so classical relativity is definitely not the only thing happening here.'''
print(comment)
                                                                                

# part (a) and (b): Calculate 'velocity' and 'gamma' arrays here. 
P = np.arange(0,1,0.02)
c =            3*10**8
v =            c*np.arange(0,1,0.02)
gamma =        1/np.sqrt(1-v**2/c**2) 


# part (c) Plot the relationship between velocity and the Lorentz factor.
plt.plot(v,gamma,'bo')
plt.xlabel('v')
plt.ylabel('gamma')
# You can set the ranges you want to plot with the following two commands:
plt.xlim(0,c)    #This sets the range of velocity values shown in the plot.
plt.ylim(0,10)   #This sets the range of gamma values shown in the plot.



# part (d) Print the gamma value associated with v = 0.98c.

#because of the spacing in the v vector, the final entry is 0.98c.0
print('gamma for muon travelling at 0.98c =', max(gamma))
# Calculate the number of muons here and print your result. 

d = 15000
v = 0.98*3*10**8

No = 5.1*10**20
t = (d/v)
To = 2.197*(10**-6)*max(gamma)

N = No*np.exp(-t/To)

print(N)

# Calculate the number of muons here, print the calculated result and the short discussion. 

d = 15000/max(gamma)
v = 0.98*3*10**8

No = 5.1*10**20
t = (d/v)
To = 2.197*(10**-6)

N = No*np.exp(-t/To)

print(N)


# Insert your calculation, plot, comment here. [25 points]

# 1.
KE_2GeV = 2 #in GeV
mc2 = 0.1*10**-10 #in GeV
gamma_2GeV = (KE_2GeV+mc2)/mc2 #in GeV

# 2.
#plot KE vs survival probability
#survival probability = No/N (final vs inital muons)
c = 3*10**8

restmass = 0.1

KE = np.arange(0,10000,0.04)

v = c*np.sqrt(1-(restmass/(KE+restmass)))

gamma = 1/(np.sqrt(1-v**2/c**2))

d = 15000/gamma


No = 5.1*10**20
t = (d/v)
To = 2.197*(10**-6)
N = No*np.exp(-t/To)


Prob = N/No

plt.plot(KE, Prob, 'bo')
plt.xlabel('KE, GeV')
plt.ylabel('Probability of reaching earth')


# 3.
#Based on the graph this looks like we will get a 90% survival probability at approximately 6000 GeV.