x = 2 + 3
x+4
def square(n: int) -> int:
    return n ** 2
square(2)
square(3)
square(15)
def orbit_kepler(p: float) -> float:
        return (p**2)**(1/3)


orbit_kepler(1)  # Earth, should return 1.0
orbit_kepler(29.660974748248961) # Saturn, should return 9.5820172
orbit_kepler(0.24084173359179098) # Mercury, should return 0.38709821
import math
math.pi
def orbit_newton(p: float, m1: float, m2: float) -> float:
    return ((((p*86400)**2)*(6.67e-11*(m1+m2))/(4*math.pi**2))**(1/3))/149000000/1000
orbit_newton(384.843, 2.0626e30, 2.986e25)  # With initial values, should return approximately 0.1230
def orbit_newton_revised(p: float, m1: float, m2: float) -> float:
        return  ((((p*86400)**2)*(6.67e-11*(m1+m2))/(4*3.14**2))**(1/3))/149000000/1000


orbit_newton_revised(28.143, 6.1659e29, 2.2693e25)  # With original values of Pi, should return approximately 0.1230
def exo_weight (w: float, m:float, r:float) :
    return (𝑤*m)/(r**2)
exo_weight (5.5, 5, 1.5)
def subjective_time (d:float, f:float):
    return (d/f)*((1-f**2)**0.5)
subjective_time (1402, .9)
