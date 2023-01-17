''' Part (i)'''



!pip install uncertainties # this installs uncertainites module to allow the propogation of error

import math

import numpy as np #this imports a library to deal with array computations

import uncertainties 

from uncertainties import ufloat

from uncertainties.umath import sin

from uncertainties.umath import acos



# we will write down the period and standard error which we will define as 0.0000001/2 = +/- 0.00000005 days, this is because the period was given as accurate to the 7th decimal place so there is an uncertaintiy of +/- 0.00000005 days



p = ufloat(4.9546416,0.00000005)



# we will  get an estimate for the mean change in flux



mean_flux_depth = 0.990 # from the plot

mean_flux_peak = 1 # from plot



# we will now estimate the standard error in the depth and peak fluxs. 

# we do this by looking at the range of values for the depth and peak ie: depth ranges between 0.989  and 0.992, and the peak ranges between 0.998 and 1.002.

# with these ranges we can estimate the standard error at each of these points ie: standard_err_depth = (0.992-0.989)/2 = 0.0015, standard_err_peak = (1.002-0.998)/2 = 0.002



standard_err_depth = (0.992-0.989)/2 # these will give standard error bounds where the true mean of the depth/peak flux will most likely be located in 

standard_err_peak = (1.002-0.998)/2 



flux_depth = ufloat(mean_flux_depth,standard_err_depth)

flux_peak = ufloat(mean_flux_peak,standard_err_peak)



deltaF = flux_peak - flux_depth





# we will now estimate the mean of the transit times Tf and Tt in days



Tf_mean = 0.027 # Tf spans from an orbital phase of -0.0135 to 0.0135 so Tf is 0.027*p days

Tt_mean = 0.034 # Tt spans from an orbital phase of -0.017 to 0.017 so Tt is 0.034*p days



# from looking at the plots we estimate the standard error (by estimating the range as well as acknowleging that our method of otaining these times (with a ruler on the screen) is in itself inaccurate) of these times to be +/- 0.001*p  days



Tf = ufloat(Tf_mean,0.001)*p

Tt = ufloat(Tt_mean,0.001)*p





# we will now calculate the impact parameter b, we will break it into pieces 



a0 = (1-(deltaF)**(0.5))**2

a1 = (sin(Tf*(math.pi)/p))/(sin(Tt*(math.pi)/p))

a2 = (1+(deltaF)**(0.5))**2





b = ((a0-(a1**2)*a2)/(1-(a1**2)))**(0.5) #impact parameter





# we will now calculate the a/Rstar ratio



a_Rstar_ratio = (((a2 - (b**2)*(1-(sin(Tt*math.pi/p)**2))))/(sin(Tt*math.pi/p)**2))**(0.5)



# we will now the stellar density



rho = ((3*math.pi)/(((p*86400)**2)*(6.67e-11)))*a_Rstar_ratio**3 # the factor of 86400 is to convert the period to seconds, and rho is in units of kg/m^3



# with the stellar density we can now find the stallar mass via the Mstar/Msun relation

rhoSun = 1408 #kg/m^3

Msun = 1.98847e30 # mass of our sun  in kg

k = 1 # for main sequence

x = 0.8 # approximatly for main sequence

Mstar_Msun = ((k**3)*(rho/rhoSun))**(1/(1-3*x))



print('We give the mass of the star as',Mstar_Msun,'solar masses')



# we will now calculate the stars radius



Rstar_Rsun = k*Mstar_Msun**x

print('We give the radius of the star as',Rstar_Rsun,'solar radii')



# we will now give the stars mass and raius in terms of SI units



Mstar = (1.98847e30)*Mstar_Msun # in Kg

Rstar = (6.95700e8)*Rstar_Rsun # in metres



print('The mass of the Star in Kgs is',Mstar,'Kg')

print('The radius of the Star in metres is',Rstar,'m')



# we will now find the orbital radius a



a = ((((p*86400)**2)*(6.67e-11)*Mstar)/(4*(math.pi**2)))**(1/3) # in metres

print('The orbital radius of the planet in metres is',a,'m')

print('The orbital radius of the planet in AU is',a/(1.495978707e11),'AU')



# now we will calculate the orbital inclination



i = acos(b*Rstar/a)



print('the orbital inclination angle in radians is given as',i,'radians')

print('the orbital inclination angle in degrees is given as',i*180/math.pi,'degrees')



# now we will find the planatery radius 



Rplanet_Rsun = Rstar_Rsun*(deltaF**(0.5))

print('the radius of the planet in solar radii is',Rplanet_Rsun,'solar radii')



Rplanet = Rplanet_Rsun*6.95700e8 # Radius of planet in metres

Rplanet_Rjup = Rplanet/(7.1492e7)



print('the radius of the planet in metres is',Rplanet,'m')

print('the radius of the planet in jovian radii is',Rplanet_Rjup,'jovian radii')



# we will now estimate the mass of the planet 

# to do this we shall estimate that the desnisty of the 'hot jupiter' is similar to that of jupitier 

# ie rho_jup = 1326 Kg/m^3

# with this we can estimate the volume of the planet as 4/3*pi*R^3



volume_planet = (4/3)*math.pi*Rplanet**3

rho_jup = 1326

mass_planet = rho_jup*volume_planet

print('we estimated the mass of the planet in kgs as',mass_planet,'kg')

print('we estimated the mass of the planet in jovian masses as',mass_planet/(1.898e27 ),'jovian masses')
''' Part (iii)'''



import matplotlib.pyplot as plt

import math

import numpy as np

# we first define our parameter vector, this will be an array with elements corresponding to the elements the same as the parameters 

# given in the assignment ie (T0,P,e,w,i,m2,a,m1) in that ordering. they will be in units given





### part (a)



para = np.array([0,5,0,math.pi/2,math.pi/4,1.4,0.05,1.4]) #these are the parameters in the order stated above



T0 = para[0]

P = para[1]

e = para[2]

w = para[3]

i = para[4]

m2 = para[5]*(1.89813e27) #this converts mass into kg

a = para[6] *(1.495978707e11) # this converts to metres

m1 = para[7]*(1.98847e30) # this converts  mass in kg





t = np.linspace(0,20,100) # time vector in days with 100 evenly spaced elements



def eccentric_anom_calc(t,t0,p,e): #this estimates the eccentric anomaly via newtons method an outline of this can be found in https://en.wikipedia.org/wiki/Kepler%27s_equation

    M = 2*math.pi*(t-t0)/p # mean anomaly 

    if e>0.8:

        E = math.pi

        n = 0

        while n<=100:

            E = E - (E-e*math.sin(E)-M)/(1-e*math.cos(E))

            n += 1

        return E

    else:

        E = M

        n = 0

        while n<=100:

            E = E - (E-e*math.sin(E)-M)/(1-e*math.cos(E))

            n += 1

        return E

    

    

eccen_anom_list = [] #this will be the list containing the eccentric anomilies for each time t

for elements in t: # this makes a vector of eccentric anomilies over the times desired 

    eccen_anom_list += [eccentric_anom_calc(elements,T0,P,e)]





eccen_anom_list = np.array(eccen_anom_list) # this turns the eccentric anomily list into an array



f = 2*np.arctan((np.tan(eccen_anom_list/2)*math.sqrt((1+e)/(1-e)))) # this obtains true anomly vector for times t





K1 = 28.4329*math.sqrt(1/(1-e**2))*m2*(1/1.89813e27)*math.sin(i)*(((m1+m2)/1.98847e30)**(-0.5))*(a/1.495978707e11)**(-0.5) #this calculates the scaled K1 parameter 



velocity = K1*(np.cos(w+f)+e*math.cos(w)) # radial velocity vector 





plt.plot(t,velocity)

plt.title('Radial velocity curve of system (a)')

plt.xlabel('Time (days)')

plt.ylabel('Radial velocity (m/s)')

plt.show()



#### part (b)



para = np.array([5,6,0.3,3*math.pi/2,math.pi/8,0.5,0.2,0.3]) #these are the parameters in the order stated above



T0 = para[0]

P = para[1]

e = para[2]

w = para[3]

i = para[4]

m2 = para[5]*(1.89813e27) #this converts mass into kg

a = para[6] *(1.495978707e11) # this converts to metres

m1 = para[7]*(1.98847e30) # this converts  mass in kg



eccen_anom_list = [] #this will be the list containing the eccentric anomilies for each time t

for elements in t: # this makes a vector of eccentric anomilies over the times desired 

    eccen_anom_list += [eccentric_anom_calc(elements,T0,P,e)]

    

eccen_anom_list = np.array(eccen_anom_list) # this turns the eccentric anomily list into an array



f = 2*np.arctan((np.tan(eccen_anom_list/2)*math.sqrt((1+e)/(1-e)))) # this obtains true anomly vector for times t





K1 = 28.4329*math.sqrt(1/(1-e**2))*m2*(1/1.89813e27)*math.sin(i)*(((m1+m2)/1.98847e30)**(-0.5))*(a/1.495978707e11)**(-0.5) #this calculates the scaled K1 parameter 



velocity = K1*(np.cos(w+f)+e*math.cos(w)) # radial velocity vector 





plt.plot(t,velocity)

plt.title('Radial velocity curve of system (b)')

plt.xlabel('Time (days)')

plt.ylabel('Radial velocity (m/s)')

plt.show()





#### part (c)



para = np.array([10,10,0.6,3*math.pi/4,math.pi/3,2.5,0.01,1]) #these are the parameters in the order stated above



T0 = para[0]

P = para[1]

e = para[2]

w = para[3]

i = para[4]

m2 = para[5]*(1.89813e27) #this converts mass into kg

a = para[6] *(1.495978707e11) # this converts to metres

m1 = para[7]*(1.98847e30) # this converts  mass in kg



eccen_anom_list = [] #this will be the list containing the eccentric anomilies for each time t

for elements in t: # this makes a vector of eccentric anomilies over the times desired 

    eccen_anom_list += [eccentric_anom_calc(elements,T0,P,e)]

    

eccen_anom_list = np.array(eccen_anom_list) # this turns the eccentric anomily list into an array



f = 2*np.arctan((np.tan(eccen_anom_list/2)*math.sqrt((1+e)/(1-e)))) # this obtains true anomly vector for times t





K1 = 28.4329*math.sqrt(1/(1-e**2))*m2*(1/1.89813e27)*math.sin(i)*(((m1+m2)/1.98847e30)**(-0.5))*(a/1.495978707e11)**(-0.5) #this calculates the scaled K1 parameter 



velocity = K1*(np.cos(w+f)+e*math.cos(w)) # radial velocity vector 





plt.plot(t,velocity)

plt.title('Radial velocity curve of system (c)')

plt.xlabel('Time (days)')

plt.ylabel('Radial velocity (m/s)')

plt.show()







#### part (d)



para = np.array([15,12,0.9,math.pi/4,math.pi/4,5,1,1.5]) #these are the parameters in the order stated above



T0 = para[0]

P = para[1]

e = para[2]

w = para[3]

i = para[4]

m2 = para[5]*(1.89813e27) #this converts mass into kg

a = para[6] *(1.495978707e11) # this converts to metres

m1 = para[7]*(1.98847e30) # this converts  mass in kg



eccen_anom_list = [] #this will be the list containing the eccentric anomilies for each time t

for elements in t: # this makes a vector of eccentric anomilies over the times desired 

    eccen_anom_list += [eccentric_anom_calc(elements,T0,P,e)]

    

eccen_anom_list = np.array(eccen_anom_list) # this turns the eccentric anomily list into an array



f = 2*np.arctan((np.tan(eccen_anom_list/2)*math.sqrt((1+e)/(1-e)))) # this obtains true anomly vector for times t





K1 = 28.4329*math.sqrt(1/(1-e**2))*m2*(1/1.89813e27)*math.sin(i)*(((m1+m2)/1.98847e30)**(-0.5))*(a/1.495978707e11)**(-0.5) #this calculates the scaled K1 parameter 



velocity = K1*(np.cos(w+f)+e*math.cos(w)) # radial velocity vector 





plt.plot(t,velocity)

plt.title('Radial velocity curve of system (d)')

plt.xlabel('Time (days)')

plt.ylabel('Radial velocity (m/s)')

plt.show()