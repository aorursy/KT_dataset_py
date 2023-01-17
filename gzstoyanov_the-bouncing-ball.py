%matplotlib inline
import matplotlib.pyplot as plt

import numpy as np
g = -9.81

y = 0.0

v = 0.0



t = 0

dt = 0.01



data = []



while t < 10:

    y += v*dt

    v += g*dt

    data.append([t, y, v]) 

    t += dt



data = np.array(data) 
data.shape
data = data.transpose()

data.shape  # t, y, v
plt.plot(data[0], data[1])

plt.xlabel("time (s)")

plt.ylabel("position (m)");
g = -9.81

y = 0.0

v = 0.0



t = 0

dt = 0.01



y_floor = -5



data = []



while t < 10:

    y += v*dt

    if y > y_floor:

        v += g*dt

    else:

        v = -v   # bounce off floor

    data.append([t, y, v]) 

    t += dt



data = np.array(data).transpose()
plt.plot(data[0], data[1])

plt.xlabel("time (s)")

plt.ylabel("position (m)");
def bouncingBall(g: 'gravity value of g(m/s2)',

                 y: 'vertical position',

                 v: 'velocity',

                 t: 'time',

                 dt: 'time step',

                 y_floor: 'floor position',

                 e: 'Coefficient of restitution',

                 interval: 'time interval',

                 title: 'name of experiment'):    

   

    data = []

    

    while t < interval:

        y += v*dt

        if y > y_floor:

            v += g*dt

        else:

            v=-e*v # bounce off floor with Coefficient of restitution - COR 

        data.append([t, y, v]) 

        t += dt

    

    data = np.array(data).transpose()

    plt.plot(data[0], data[1])

    plt.title(title)

    plt.xlabel("time (s)")

    plt.ylabel("position (m)")
bouncingBall(-9.81, 0, 0, 0, 0.01, -5, 1, 10, "e(Coefficient of restitution - COR) = 1")
bouncingBall(-9.81, 0, 0, 0, 0.01, -5, 0, 10, "e = 0")
bouncingBall(-9.81, 0, 0, 0, 0.01, -5, 2, 10, "e = 2")
bouncingBall(-9.81, 0, 0, 0, 1, -5, 2, 1400000, "e = 2")
bouncingBall(-9.81, 0, 0, 0, 0.05, -5, 0.61, 5, "Cricket Ball - e = 0.61")
bouncingBall(-9.81, 0, 0, 0, 0.05, -5, 0.636, 5, "Hockey Ball - e = 0.636")
bouncingBall(-9.81, 0, 0, 0, 0.05, -5, 0.794, 6, "Table Tennis Ball - e = 0.794")
bouncingBall(-9.81, 0, 0, 0, 0.05, -5, 0.893, 11, "Golf Ball - e = 0.893")