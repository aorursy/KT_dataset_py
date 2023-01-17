import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%pylab inline



from mpl_toolkits.mplot3d import Axes3D



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
impacts = pd.read_csv('../input/impacts.csv')

orbits = pd.read_csv('../input/orbits.csv')

impacts.info()
orbits.info()
sns.lmplot(x='Asteroid Velocity', y='Asteroid Magnitude', hue='Cumulative Impact Probability',

           legend=False, data=impacts, fit_reg=False)

plt.title('Pink is highly probable and Blue is not')
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')



x, y, z = (impacts['Asteroid Velocity'],

           impacts['Asteroid Magnitude'],

           impacts['Cumulative Impact Probability'])

ax.scatter(x, y, z)

ax.view_init(30, 80)

plt.xlabel('Velocity')

plt.ylabel('Magnitude')

plt.title('Cumulative Impact Probability')
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')



x, y, z = (impacts['Asteroid Velocity'],

           impacts['Asteroid Magnitude'],

           impacts['Possible Impacts'])

ax.scatter(x, y, z)

ax.view_init(30, 80)

plt.xlabel('Velocity')

plt.ylabel('Magnitude')

plt.title('Possible Impacts')
fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')



x, y, z = orbits['Orbit Axis (AU)'], orbits['Orbit Eccentricity'], orbits['Orbit Inclination (deg)']

ax.scatter(x, y, z)

ax.view_init(30, 80)

plt.xlabel('Orbit Axis')

plt.ylabel('Orbit Eccentricity')

plt.title('Orbit Inclination')
o = orbits['Orbit Axis (AU)'].copy()

mask = np.abs(o - o.mean()) < (o.std() * 2.9)

O = orbits.loc[mask].copy()



fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(111, projection='3d')



x, y, z = O['Orbit Axis (AU)'], O['Orbit Eccentricity'], O['Orbit Inclination (deg)']

ax.scatter(x, y, z)

ax.view_init(10, 80)

plt.xlabel('Orbit Axis')

plt.ylabel('Orbit Eccentricity')

plt.title('Orbit Inclination')