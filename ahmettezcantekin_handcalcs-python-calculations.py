#Install necessary libraries

!pip install handcalcs

!pip install forallpeople
#Import handcalcs

import handcalcs.render
%%render

#Basic Usage

a = 2

b = 3

c = 2*a + b/3
%%render

a = 3

b = 4

c = a*b*2

c
import handcalcs.render

from math import sqrt, pi
%%tex

#This will produce a LaTeX output as follows.

a = 2 / 3 * sqrt(pi)
from math import sqrt, sin, pi
%%render

d = sqrt(a) + sin(b) + pi + sqrt(a)  # this is d

e = sin(b) + pi + sqrt(a) + sin(b) + pi # this e
from handcalcs.decorator import handcalc



@handcalc(jupyter_display = True)

def some_cal(a,b):

    d = sqrt(a) + sin(b) + pi + sqrt(a)  # this is d

    e = sin(b) + pi + sqrt(a) + sin(b) + pi # this is e

    return locals()
some_cal(1,2)
#Use the # Long or # Short comment tags to override the length check and display the calculation in the "Long" format or the "Short" format for all calculations in the cell. e.g
import handcalcs.render

from math import sqrt, sin, asin, pi
%%render

f= d/ a+b # Comment

g=d*f / a # Comment

d=sqrt(a / b) + asin(sin(b / c)) + (a/b)**(0.5) + sqrt((a*b + b*c)/(b**2)) + sin(a/b)  # Long
%%render

# Short

f= d/ a+b # Comment

g=d*f / a # Comment

d=sqrt(a / b) + asin(sin(b / c)) + (a/b)**(0.5) + sqrt((a*b + b*c)/(b**2)) + sin(a/b)  # short format
%%render

# Long

f= d/ a+b # Comment

g=d*f / a # Comment

d=sqrt(a / b) + asin(sin(b / c)) + (a/b)**(0.5) + sqrt((a*b + b*c)/(b**2)) + sin(a/b)  # long format
%%render

# Parameters

a=23

b=43

c=2
%%render

r=sqrt(a**2 +b**2)

x=(-b + sqrt(b**2 -4*a*c))/(2*a)
%%render

#Symbolic

r=sqrt(a**2 +b**2)

x=(-b + sqrt(b**2 -4*a*c))/(2*a)
%%render

# Parameters

r

x
import handcalcs.render

import forallpeople

%env 'structural'
%%render

# Parameters

# phi = 0.9

# Z_x = 234e3 * mm**3

# F_y = 400 + MPa



a_x=23

b_y_d=234

A_box_red_small=23940
%%render

alpha=0.9

beta_x=23.1

Omega_eta_nu=234
%%render

a=23

b=54

c=min(a,b)
def square(x):

    return x**2
%%render

k_x=square(a)
%%render

lamb=0.734 # This is an approximate scaling factor

a=23.4

b=6*a

f= lamb * a + b/a
%%render

x_y=5
%%render

if x_y<10: b=10*x_y

elif 10<= x_y <=60: b= x_y**2
%%render

x_y=50
%%render

if x_y<10: b=10*x_y

elif 10<= x_y <=60: b= x_y**2