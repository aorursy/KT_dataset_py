#Lets import packages.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
#Read impacts csv into dataframe.
neo = pd.read_csv("../input/impacts.csv", parse_dates=["Period Start","Period End"])
#Lets see what columns we have.
neo.columns
#Lets also take a look at the first few rows of data.
neo.head()
#The question I will attempt to answer is "Which possible asteroid impact would 

#be the most devastating, given the asteroid's size and speed?" The most devastating 

#impact would be the one that would impart the largest force. The equation F=ma 

#(F is force, m is mass, and a is acceleration) could be used to estimate the force 

#an asteroid impact would have. Unfortunately, determining the rate of acceleration 

#(deceleration in this case) an asteroid would have when it impacted the earth is 

#difficult to estimate, let alone compute. An equation that is related to F=ma is 

#momentum, p=mv (p is momentum, m is mass, and v is velocity). Momentum is the force 

#an object has when it is moving. In addition, Newtons third law of motion states 

#that for every action, there is an equal and opposite reaction. When an object 

#strikes another, the object that was struck imposes an equal and opposite force on 

#the striking object. The momentum of an asteroid could be used as an indicator of 

#the force required to stop it. Before we can calculate an asteroids momentum, we 

#need to first calculate its mass. In order to calculate its mass, we need its volume 

#and density. I will assume the asteroid is spherical for simplicity when calculating 

#volume. I will also assume the worst case scenario for the density of the asteroid. 

#The most dense asteroid discovered has a density of 3.4e+12 kg/km^3 

#(https://www.newscientist.com/article/dn21101-most-pristine-known-asteroid-is-denser-than-granite/).

#Volume will be m^3. Density will be kg/m^3. Mass will be kg. Velocity will be m/s. Momentum will be kg m/s.
neo['Volume']=(4/3)*((neo['Asteroid Diameter (km)']*500)**3)*np.pi
neo['Mass']=neo['Volume']*(3400)
neo['Momentum']=neo['Mass']*neo['Asteroid Velocity']*1000
#The asteroid with the largest momentum when it reaches the earth's atmosphere is...

neo[(neo['Momentum']==max(neo['Momentum']))]
#Another characteristic of a moving object is kinetic energy. Its formula is E=0.5*m*v^2.

#Kinetic energy would be another good descriptor of the damage an asteroid would do since

#since kinetic energy itself is directly proportional to how much damage it inflicts when

#striking something. E will be kg m^2/s^2 or Joules.
neo['Kinetic Energy']=0.5*neo['Mass']*((neo['Asteroid Velocity']*1000)**2)
#The asteroid with the largest kinetic energy when it reaches the earth's atmosphere is...

neo[(neo['Kinetic Energy']==max(neo['Kinetic Energy']))]
#We can also convert Joules into Tons of TNT. The US nuclear detonation was Castle Bravo

#which had a explosive yield of 15 Mega Tons. The asteroid with the largest kinetic energy

#would have a yield of 928000 Mega Tons when it reached the atmosphere.

max(neo['Kinetic Energy'])*(2.39006*10**-10)
#Alternatively, the asteroid with the smallest kinetic energy would have

#a yield have 2 Tons.

min(neo['Kinetic Energy'])*(2.39006*10**-10)
#Log plot of asteroid kinetic energy for objects in the impact csv.

neo.sort_values('Kinetic Energy',inplace=True)

ke = neo['Kinetic Energy'].values

plt.plot(ke,'g.')

plt.xticks([])

plt.yscale('log')

plt.grid(True)