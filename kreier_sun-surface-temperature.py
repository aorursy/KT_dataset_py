m = 30.0      # mass of the water

d = 0.022     # diameter of the test tube

h = 0.1       # height of the illuminated/heated water

t = 300.0     # 5 minutes or 300 seconds illumination time

c = 4.184     # specific heat capacity of water



deltaT =  float(input("Enter the measured temperature change in Kelvin: "))
I_earth = m*c*deltaT/(d*h*t)

     

print(str(int(I_earth)) + ' W/m² is the intensity of the sun on earths surface in Ho Chi Minh City.')
r_AU  = 1.495978707E11  # Astronomical Unit - average distance sun-earth

r_sun = 6.955E8         # average radius of the sun



I_sun = I_earth * r_AU**2 / r_sun**2



print(str(int(I_sun)) + ' W/m² is the intensity on the surface of the sun.')
k = 5.67E-8  # Stefan-Boltzmann constant



T = (I_sun / k)**(1/4)



print(str(int(T)) + ' K is the surface temperature of the sun.')
m = 30                 # mass of water

c = 4.184              # specific heat capacity of water

dT = 5.9               # temperature change of the water

t = 300                # time of exposure in the sun

A = 0.0022             # area of irradiation

r_AU = 1.496E11        # Astronomical unit - average distance earth-sun

r_sun = 6.955E8        # average radius sun

sigma = 5.67E-8        # Stefan-Boltzmann constant



T = (m*c*dT/(t*A*sigma)*r_AU**2/r_sun**2)**(1/4)



print(T)
error_percent = ( 1/30 + 0.001/0.022 + 0.001/0.1 + 0.1/5.9 ) / 4



print(error_percent)

print(T * error_percent)