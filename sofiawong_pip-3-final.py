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



#arange function

t = np.arange(0,20,0.5)

x = -t/5

arange_array=np.exp(x)

plt.plot(x,arange_array,color='red')



plt.xlabel('t')    # This labels the x axis

plt.ylabel('value')

plt.title('function between t=0 and t=20')   # Title for the graph

plt.grid(True)                   # This enables that the grid is displayed

plt.show()



#linspace function

t_2 = np.linspace(0,20,50)

x_2 = -t_2/5

linspace_array=np.exp(x_2)

plt.plot(x_2,linspace_array,color='blue')



plt.xlabel('t')    # This labels the x axis

plt.ylabel('value')

plt.title('function between t=0 and t=20')   # Title for the graph

plt.grid(True)                   # This enables that the grid is displayed

plt.show()

plt.plot(x,arange_array,color='black',label='arange_exp(x)') 

plt.plot(x,arange_array,'o',color='red',label='actual points for arange function')

plt.axis([-4,0,0,1])

plt.legend()

plt.show()



plt.plot(x_2,linspace_array,color='black',label='linspace_exp(x)')

plt.plot(x_2,linspace_array,'o',color='blue',label='actual points for linspace function')

plt.axis([-4,0,0,1])

plt.legend()

plt.show()
# Distance travelled by non-relativistic muons 

# Calculate the distance and comment on your result here. 



# Calculation



T_0 = 2.197*10**-6

c = 2.99792458*10**8

x = T_0*0.98*c  # Insert your code here.



# Print your comment here.



print('The muon travels', x , 'metres.')

print('')

print('Our comment on the implication of this result:')

# Calculate the number of muons reaching sea level on Earth and print your comment here.



# (a)

t = 15*10**3/(0.98*c)

N_0 = 5.1*10**20

N = N_0*np.exp(-t/T_0)



print('The number of muons reach sea level is', N)

print('')

                                                                 
# part (a) and (b): Calculate 'velocity' and 'gamma' arrays here. 



c = 2.99792458*10**8   # Insert your code here.

step = 4.5*10**6

velocity = np.arange(0,c,step)    # Insert your code here.

gamma = 1/(1-velocity**2/c**2)**0.5       # Insert your code here.





# part (c) Plot the relationship between velocity and the Lorentz factor.

plt.plot(velocity,gamma,color='black',label='gamma & velocity graph')

plt.plot(velocity,gamma,'o',color='red',label='actual points')

plt.xlim(0,c)

plt.ylim(0,10)

plt.legend()

plt.show()



# You can set the ranges you want to plot with the following two commands:

# plt.xlim(0,c)    This sets the range of velocity values shown in the plot.

# plt.ylim(0,10)   This sets the range of gamma values shown in the plot.





# part (d) Print the gamma value associated with v = 0.98c.

c = 2.99792458*10**8 

v = 0.98*c

gamma_1 = 1/(1-v**2/c**2)**0.5

print(gamma_1)

# Calculate the number of muons here and print your result. 

T_0 = 2.197*10**-6

c = 2.99792458*10**8 

v = 0.98*c

gamma_1 = 1/(1-v**2/c**2)**0.5

t_prop = t/gamma_1

N_0 = 5.1*10**20

N = N_0*np.exp(-t_prop/T_0)



print(N)



# Calculate the number of muons here, print the calculated result and the short discussion. 

T_0 = 2.197*10**-6

c = 2.99792458*10**8 

v = 0.98*c

L_0 = 15*10**3

t = 15*10**3/(0.98*c)

gamma_1 = 1/(1-v**2/c**2)**0.5

L = L_0/gamma_1

t_0 = L/v



N_0 = 5.1*10**20

N = N_0*np.exp(-t_0/T_0)



print(N)





# Insert your calculation, plot, comment here. [25 points]



# 1.

m = 0.1*10**9

c = 2.99792458*10**8 

gamma = 1/(1-v**2/c**2)**0.5

E_k = 2*10**9

E_k = (gamma-1)*m*c**2



v = (c**2-c**2/(E_k/(m*c**2)+1)**2)**0.5



print('a) the velocity is',v,'m/s')

t = 15*10**3/v

t_0 = t/gamma

print('the time dilation factor is',t_0, 's')



# 2.



t = 15*10**3/v

t_0 = t/gamma

N_0 = 5.1*10**20

T_0 = 2.197*10**-6

N = N_0*np.exp(-t_0/T_0)

t_0 = np.arange(0,50,5)





S_prob = np.exp(-t_0/T_0)







plt.plot(t_0,S_prob,color='red')

plt.xlabel('t')    # This labels the x axis

plt.ylabel('survival probabilities')

plt.title('Survival Curve')   # Title for the graph

plt.grid(True)                   # This enables that the grid is displayed

plt.show()







# 3.


