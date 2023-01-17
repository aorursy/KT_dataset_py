# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import math
import numpy as np
pi = math.pi
e= math.e

#HVAC STATS
flow = 4800 #CFM
cooling = 60000 #BTU/hr

#UNIT D-101
k_insulation = 0.030 #BTU/(hr*ft*F)
volume = 6673 #ft^3
length = 35 #ft, length of tangent line
r_inner = 6.93 #ft, approximation
r_outter = r_inner + 6/12 #+ thickness of insulation

##CALCULATE SURFACE AREA
area_side = length*2*r_outter*pi
area_sphere = 4*pi*r_outter**2
area_total = (area_side+area_sphere)/2

area_outter = 2*pi*length*r_outter

#HEAT SOURCES
sun = 327 #BTU/(ft^2*hr)
person = 300 #BTU/hr
n = 3 #number of people inside vessel
heat_flux = sun

heat_rate = heat_flux*area_outter
print("the heat rate is", "{:e}".format(heat_rate),"BTU/hr")

deltaT = heat_rate*np.log(r_outter/r_inner)/(2*pi*k_insulation*length)
print("The temperature difference is", deltaT,"F")



net_rate = person*n-cooling
net_rate = abs(net_rate)
print(net_rate,"BTU/hr","\n")




#TEMPERATURE FROM AC
hs = 60000 #BTU/hr
q = 2000 #cfm
deltaT = hs/(1.08*q) #F
T_initial = 90 #degF
T_final = T_initial-deltaT #degF, temperature output of AC 


T_desired = 70 #degF
Tk_initial = ((T_initial)-32)*5/9+273.15
Tk_final = ((T_desired)-32)*5/9+273.15
deltaTk = Tk_initial-Tk_final


#deltaTk = ((deltaT)-32)*5/9+273.15 #K
print("The change in temp in kelvin is",deltaTk,"\n")


volume = 6673/35.315 #m^3
pressure = 101325 #Pa
#T_initialk = 305.372 #K
R = 8.314 #J/mol*K
n = pressure*volume/(R*Tk_initial) #mol air 
Cp = 7*R/2 #J/mol*K
print(deltaTk,"nnnnn")
Q = Cp*n*deltaTk #joules

print(volume)
print(Q,"Joules")

Q = Q/1055 #BTU/hr
print("A total of",round(Q),"BTUs needed to change the temp from",T_initial,"F to",T_desired,"F")

time = Q/cooling #hours
print(round(time,2),"hours minimum to change tempfrom",T_initial,"F to",T_desired,"F")
print(round(time*60),"minutes minimum to change tempfrom",T_initial,"F to",T_desired,"F")

time = Q/net_rate #BTU/hr with 3 people in there from the start
print("With 3 people in from the start",round(time,2),"hours")
print("With 3 people in from the start",round(time*60,2),"minutes")


diameter = 2 #ft
area = pi*diameter**2/4
flow = 2000 #ft^3/min
velocity = flow/area
print(velocity)
velocity = velocity*10*5280/3600
print(velocity)
