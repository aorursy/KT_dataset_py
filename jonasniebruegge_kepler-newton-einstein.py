x = 2 + 3
x+4
def square(n: int) -> int:
    return n ** 2
square(2)
square(3)
square(15)
def orbit_kepler(p: float) -> float:
    return p**(2/3)
    

orbit_kepler(1)  # Earth, should return 1.0
orbit_kepler(29.660974748248961) 
orbit_kepler(0.24084173359179098) # Mercury, should return 0.38709821
import math
math.pi
def orbit_newton(p: float, m1: float, m2: float) -> float:
    g=6.67e-11
    per=(p*86400)**(2)
    pie=4*math.pi**2
    sick=g*(m1+m2)*(per)/(pie)
    cube=sick**(1/3)
    return cube/1000/149000000
                            

orbit_newton(28.143, 6.1659e29, 2.2693e25)  # With initial values, should return approximately 0.1230
orbit_newton(11.186,2.4459e29,7.5852e24)
orbit_newton(289.860,1.9293e30,3.1532e26)
orbit_newton(197.800,1.4917e30,4.891e25)
orbit_newton(25.631,7.956e29,4.359e25)
orbit_newton(66.870,6.1659e29,2.2693e25)
orbit_newton(384.843,2.0626e30,2.986e25)
def orbit_newton_revised(p: float, m1: float, m2: float) -> float:
    g=6.67e-11
    per=(p*86400)**(2)
    pie=4*3.14**2
    sick=g*(m1+m2)*(per)/(pie)
    cube=sick**(1/3)
    return cube/1000/149000000

orbit_newton_revised(28.143, 6.1659e29, 2.2693e25)  # With original values of Pi, should return approximately 0.1230
def exo_weight(we,mp,rp):
    weight_mass=we*mp
    raduis_planet=rp**2
    weight_on_planet=weight_mass/raduis_planet
    return weight_on_planet
exo_weight(5.5,1.54,3.709)
exo_weight(5.5,17.09,2.39)
exo_weight(5.5,7.3,2.43)
exo_weight(5.5,5,1.5)
def subjective_time(f,d):
    time_back_home=d/f
    inside_f=1-(f**2)
    root=inside_f**(1/2)
    time_travelor=root*time_back_home
    return time_travelor
subjective_time(.5,23.6)