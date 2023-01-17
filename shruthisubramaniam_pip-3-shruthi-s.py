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
myarray = np.array(range(20))  # creates an array with 200 elements from 0 to 199

myarray1 = myarray/20*4*np.pi   # changes the range to go from 0 to 4pi

myarray2 = np.sin(myarray)      # stores the results of sin(x) in myarray2

print(myarray1)
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



# my code that makes two arrays and plot the function exp(-t/5) between t=0 and t=20.



time = np.linspace(0,20,1000)

y = np.exp(-time/5)

plt.plot(time,y,color='red',label='e^(-t/5)')

plt.xlabel('Time') # this is a part of the code that titles the x axis as time

plt.ylabel('e^(-t/5)') # this is a part of the code that titles the y axis as e^(-t/5)

plt.title('Graph that shows e^(-t/5) over time') # this is the title of the graph.

plt.legend()

plt.show()





# Distance travelled by non-relativistic muons 

# Calculate the distance and comment on your result here. 



T_0 = 2.197*10**(-6)  #here T_0 is 2.197 microseconds. 

c = 3*10**8 # c is considered to be 3*10^8 m/s as this is the speed of light. 

v = 0.98*c



# Calculation



x = v*T_0   # Insert your code here.



# Print your comment here.



print('The muon travels', x , 'metres.')

print('')

print('Our comment on the implication of this result: On the basis of Newtonian motion, there is no acceleration and the time span that the muon travels for is T_0. This is due to the fact that the muon travels at a speed that is 98% the speed of light. Therefore, the equation s = ut + 0.5at^2 enables us to obtain an answer for the distance that the muon travels. In the equation s = ut + at^2, u is considered the intial speed, a is consided the acceleration and t is considered the time. ')

# Calculate the number of muons reaching sea level on Earth and print your comment here.



# (a)



T_0 = 2.197*10**(-6) #here T_0 is 2.197 microseconds.

c = 3*10**8 # c is considered to be the speed of light and the speed of light is 3*10^8 m/s.

v = 0.98*c # v is considered to be the speed of the muons and this is considered to be 98% the speed of light hence 0.98*c. 

intial_muons = (5.1*(10**20))/60 # the intial_muons is an approximation of the number of muons created in the atmosphere per second.  

distance_travelled = 15*(10**3) # To arrive at sea level, muons travelled 15 km which is 15*10^3 meters. 

time_taken = distance_travelled/v # This equation for the time taken for the muons to travel to sea level assuming that there is 0 acceleration. 

muon_num_sl = intial_muons*np.exp(-time_taken/T_0) # muon_num_sl = the number of muons that arrived at sea level. Using the equation N=N0e^(-t/T0), we can calculate the number of muons that arrive at sea level per second. 

print('part a: per minute, approximately',round(muon_num_sl*60),'muons are assumed to attain sea level')



# (b)

E_surfacearea = 5.1*(10**14) # Here E_surfacearea is the earth's surface area. The earth's total surface area is approximately 5.1*10^14.

workedout_flux_result = (muon_num_sl*60)/E_surfacearea #the workedout_flux_result is the amount of calculated flux of the muons heading towards sea level per min. 

print('part b:', workedout_flux_result , 'is the flux calculated per minute per square meter. This is comparitively lower than the specified value of the flux  10^4 per minute per square meter. This is due to the fact that our presnt calculation accounts for non relativistic muons rather than relativistic muons.')





                                                                                
# part (a) and (b): Calculate 'velocity' and 'gamma' arrays here. 



c = 3*10**8           # speed of light.

velocity = np.arange(0, c, 600000)     # Insert your code here.

gamma = 1/((1-((velocity**2)/(c**2)))**0.5)       # Insert your code here.





# part (c) Plot the relationship between velocity and the Lorentz factor.



# You can set the ranges you want to plot with the following two commands:

plt.xlim(0,c) #The causes tha range of velocity values displayed in the plot to be shown.

plt.ylim(0,10) #the causes the range of gamma values displayed in the plot to be shown. 



plt.plot(velocity,gamma,color='green',label='Lorentz factors')

plt.xlabel('Velocity') #this part of the code causes the x axis to be titled as velocity.

plt.ylabel('Lorentz factors') #This part of the code causes the y axis to be titled as Lorentz factors.

plt.title('A graph that displays Lorentz factor against velocity') #this is the title of the graph.

plt.legend()

plt.show()



# part (d) Print the gamma value associated with v = 0.98c.



print('0.98c as the Lorentz factor is', 1/((1-(((0.98*c)**2)/(c**2)))**0.5),'.')



# Calculate the number of muons here and print your result. 



T_0 = 2.197*10**(-6) #here T_0 is 2.197 microseconds.

c = 3*10**8 # c is considered to be the speed of light and the speed of light is 3*10^8 m/s.

v = 0.98 * c # v is considered to be the speed of the muons and this is considered to be 98% the speed of light hence 0.98*c. 

gamma = 1/((1-((v**2)/(c**2)))**0.5)

intial_muons = (5.1*(10**20))/60 # the intial_muons is an approximation of the number of muons created in the atmosphere per second.

distance_travelled = 15*(10**3) # To arrive at sea level, muons travelled 15 km which is 15*10^3 meters.

time_taken = distance_travelled/v # This equation for the time taken for the muons to travel to sea level assuming that there is 0 acceleration.

tau = time_taken/gamma #This code is based on the equation t = tau * gamma

muon_num_sl = intial_muons*np.exp(-tau/T_0) #muon_num_sl = the number of muons that arrived at sea level. Using the equation N=N0e^(-tau/T0), we can calculate the number of muons that arrive at sea level per second.

E_surfacearea = 5.1*(10**14) #Here E_surfacearea is the earth's surface area. The earth's total surface area is approximately 5.1*10^14.

print('The impact of time dialtion can be inspected through the view point of a on looker at sea. from the on looker at seas viewpoint we can see the number of muons appearing at sea level is 5.018541894617659e+19 per minute. In other words this is a flux of 98402.7822470509 per square meter per minute')

















# Calculate the number of muons here, print the calculated result and the short discussion. 

T_0 = 2.197*10**(-6) #here T_0 is 2.197 microseconds.

c = 3*10**8 # c is considered to be the speed of light and the speed of light is 3*10^8 m/s.

v = 0.98 * c # v is considered to be the speed of the muons and this is considered to be 98% the speed of light hence 0.98*c. 

gamma = 1/((1-((v**2)/(c**2)))**0.5)

intial_muons = (5.1*(10**20))/60 # the intial_muons is an approximation of the number of muons created in the atmosphere per second.

distance_travelled_intial = 15*(10**3) # To arrive at sea level, muons travelled 15 km which is 15*10^3 meters.

distance_travelled = distance_travelled_intial/gamma # This equation calculates the contracted length.  

time_taken = distance_travelled/v # This equation for the time taken for the muons to travel to sea level assuming that there is 0 acceleration.

muon_num_sl = intial_muons*np.exp(-time_taken/T_0) # muon_num_sl = the number of muons that arrived at sea level. Using the equation N=N0e^(-t/T0), we can calculate the number of muons that arrive at sea level per second.

print('part a: when we are too think about the impact of the length contraction from the view point of the muon, 5.018541894617659e+19 number of muons arrive at sea level per minute. In other words this is a flux of 98402.7822470509. per meter square per minute.')

print('part b: We can cpmprehend from the answer we obtain from the view point of the muon when considering the length contraction that we obtain the same result as we we calculate from the viewpoint of the observer at sea level when we are taking into account the time dilation. We see that in terms of space time, since the space time aspect sustained, we see that the length contraction (space) produces the same impact as the time dilation (time). However, in comparison to the classical case, We are able to see that we obtain different result mainly due to the fact that the other two cases consider relativity. Hence we obtain much larger results for the classical case')









# Insert your calculation, plot, comment here. [25 points]



# 1.



rest_energy = 0.1*10**9 #rest energy = rest mass

ke = 2*10**9 #2 GeV's kinetic energy.

gamma = (ke/rest_energy)+1 #here the Lorentz fact is deemed as gamma.

print('part a', gamma, 'is the time dilation factor, the Lorentz factor')



# 2.

T_0 = 2.197*10**(-6) #here T_0 is 2.197 microseconds.

intial_muons = (5.1*(10**20))/60 # the intial_muons is an approximation of the number of muons created in the atmosphere per second.

c = 3*10**8 #the is the speed of light. 

possible_energies = np.linspace(1*10**8,30*10**9,300) # this takes a range of energies for investigation.

possible_factors = (possible_energies/rest_energy)+1 # Obtaining Lorentz factor given each energy value.

speed = ((1-((1/possible_factors)**2))*(c**2))**0.5 # An equation to obtain the reult for the speed of the Lorentz factor. 

distance_travelled = 15*(10**3) #15 km which is 15*10^3 meters.

time = distance_travelled/speed #using equation s = ut +0.5at^2 to find time where a = 0.

true_time = time/possible_factors #using equation lorentz factor *true_time = time.

muon_num_sl = intial_muons*np.exp(-true_time/T_0) # muon_num_sl = the number of muons that arrived at sea level. Using the equation N=N0e^(-true_time/T0), we can calculate the number of muons that arrive at sea level per second.



plt.plot(possible_energies,muon_num_sl/intial_muons,color='purple',label='proportion of muons arriving at sea level')

plt.xlabel('Kinetic energy (eV)') #this code gives a title for the x axis.

plt.ylabel('number of muons arriving at sea level') #This code gives a title for the y axis.

plt.title('part b: a graph that shows the number of nuons arriving at sea level against the kinetic energy') #title for the graph.

plt.legend()

plt.xlim(0,30*10**9)

plt.ylim(0,1)

plt.show



# 3.

time_for_point9N = -T_0*np.log(0.9) # This is a code that finds the true time.

speed_for_point9N = distance_travelled/time_for_point9N # This is a code that helps to calculate the speed. 

factor = 1/((1-(1/speed_for_point9N/(c**2)))**0.5) #this is for calculating the lorentz factor.

total_energy = 0.5*speed_for_point9N**2 *(rest_energy/c**2) + rest_energy

print('part c: in order for 90% of the muons to appear at sea, the minimum amount of energy that is required is 2332989262352.41 eV (the result has been worked out). Through the graph we are able to see that the total kinectic energy = kinetic energy + rest mass. This total kinectic energy is 20.1 Gev and this is close to the result that has been worked out.')


