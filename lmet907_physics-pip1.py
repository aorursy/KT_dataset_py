# import standard python libraries for plotting and calculations

import matplotlib.pyplot as plt

import numpy as np

import scipy.constants as constants

from textwrap import dedent
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



def task2():

    x_data = np.linspace(0, 20)

    y_data = np.exp(-x_data / 5)

    plt.plot(x_data, y_data, label = r'$\exp \left( \frac{-t}{5} \right)$')

    plt.legend()

    plt.show()



task2()
def task3():

    muon_speed = 0.98 * constants.c

    muon_mean_lifetime = 2.197E-6

    x = muon_speed * muon_mean_lifetime



    print(dedent('''\

        The muon travels {:.1f} metres.

        

        This would imply that every 600 metres we should

        expect to lose most of our muons.\

    '''.format(x)))

    

task3()
# Calculate the number of muons reaching sea level on Earth and print your comment here.



def task4():

    distance = 15E3

    muon_speed = 0.98 * constants.c

    muon_mean_lifetime = 2.197E-6

    muon_necessary_lifetime = distance / muon_speed

    muon_generation_rate = 5.1E20

    muon_survival_rate = muon_generation_rate * np.exp(-muon_necessary_lifetime / muon_mean_lifetime)

    

    earth_area = 510.1E12

    muon_flux = muon_survival_rate / earth_area

    

    print(dedent('''\

        We would expect that of the {:.2e} muons generated

        per minute only {:.2e} make it to sea level. This means

        a flux of {:.2e} muons per square metre per minute.

        This is much less than the measured 1e4 muons detected

        per square metre per minute, so something is up.\

    ''').format(muon_generation_rate, muon_survival_rate, muon_flux))



task4()
def task5():

    lorentz = lambda v: 1 / np.sqrt(1 - (v / constants.c) ** 2)



    v_data = np.linspace(0, constants.c, endpoint = False)

    plt.plot(v_data, lorentz(v_data))

    plt.title('Lorentz factor dependency on velocity')

    plt.xlabel('velocity (m/s)')

    plt.ylabel('Lorentz factor')

    plt.show()

    

    print('The Lorentz factor for a muon travelling at 0.98c is {:.2f}'.format(lorentz(0.98 * constants.c)))

    

task5()
def task6():

    lorentz = lambda v: 1 / np.sqrt(1 - (v / constants.c) ** 2)

    

    distance = 15E3

    muon_speed = 0.98 * constants.c

    muon_mean_lifetime = 2.197E-6

    muon_dilated_mean_lifetime = lorentz(muon_speed) * muon_mean_lifetime

    muon_travel_time = distance / muon_speed

    muon_generation_rate = 5.1E20

    muon_survival_rate = muon_generation_rate * np.exp(- muon_travel_time / muon_dilated_mean_lifetime)

    

    earth_area = 510.1E12

    muon_flux = muon_survival_rate / earth_area

    

    print(dedent('''\

        We would expect that of the {:.2e} muons generated

        per minute only {:.2e} make it to sea level. This means

        a flux of {:.2e} muons per square metre per minute,

        very close to the measured 1e4.\

    ''').format(muon_generation_rate, muon_survival_rate, muon_flux))



task6()
def task7():

    lorentz = lambda v: 1 / np.sqrt(1 - (v / constants.c) ** 2)

    

    distance = 15E3

    muon_speed = 0.98 * constants.c

    contracted_distance = distance / lorentz(muon_speed)

    muon_mean_lifetime = 2.197E-6

    muon_travel_time = contracted_distance / muon_speed

    muon_generation_rate = 5.1E20

    muon_survival_rate = muon_generation_rate * np.exp(- muon_travel_time / muon_mean_lifetime)

    

    earth_area = 510.1E12

    muon_flux = muon_survival_rate / earth_area

    

    print(dedent('''\

        We would expect that of the {:.2e} muons generated

        per minute only {:.2e} make it to sea level. This means

        a flux of {:.2e} muons per square metre per minute.

        This is the same value as that calculated in the

        earth's frame of reference. It is much higher than the

        classical value, which is to be expected since in the

        muons reference frame the distance travelled is less.\

    ''').format(muon_generation_rate, muon_survival_rate, muon_flux))



task7()
def task8():

    distance = 15E3

    muon_rest_energy = 0.1

    muon_mean_lifetime = 2.197E-6

    

    print("The time dilation factor is {}".format(2 / muon_rest_energy + 1))



    def survival_probability(muon_kinetic_energy):

        lorentz_gamma = muon_kinetic_energy / muon_rest_energy + 1

        muon_travel_time = distance / constants.c / np.sqrt(lorentz_gamma ** 2 - 1)

        return np.exp(- muon_travel_time / muon_mean_lifetime)



    ke_data = np.linspace(0.0001, 5)

    plt.plot(ke_data, survival_probability(ke_data))

    plt.title('Survival probability of muons of kinetic energies')

    plt.xlabel('Kinetic energy (GeV)')

    plt.ylabel('Survival probability')

    plt.show()

    

    probability = 0.9

    minimum_energy = muon_rest_energy * np.sqrt(1 + (distance / constants.c / muon_mean_lifetime / np.log(probability)) ** 2)

    print(dedent('''\

        The minimum total energy for a {} probability

        of survival is {:.2e} GeV\

    '''.format(probability, minimum_energy)))

    

task8()