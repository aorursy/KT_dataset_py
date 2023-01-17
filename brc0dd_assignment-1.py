import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



t = np.arange(0, 10, 0.1)

y_s = np.sin(t)

y_c = np.cos(t)



plt.plot(t, y_s, t, y_c)



plt.xlabel('Time -->')

plt.ylabel('Amplitude -->')

plt.title('Plot of sin(t) and cos(t)')



plt.legend(['sin(t)', 'cos(t)'])

plt.grid(True, which='both')

plt.axhline(y=0, color='k')



plt.show()
amplitude = pd.DataFrame(data=[t, y_s, y_c],

                         index=['time', 'amplitude1', 'amplitude2']).T

amplitude.head()
amplitude.to_csv('amplitude.csv', index=False)
read_data = pd.read_csv('amplitude.csv')

read_data.head()
fig, ax = plt.subplots()



read_data.plot(ax=ax, x='time', y='amplitude1', grid=True, label="sin(t)")

read_data.plot(ax=ax, x='time', y='amplitude2', grid=True, label="cos(t)")



ax.set_ylabel('amplitude')

ax.set_title('Plot of sin(t) and cos(t)')

ax.axhline(y=0, color='k')



plt.savefig('amplitude.png')

plt.show()
import os



for dirname, _, filenames in os.walk('.'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from IPython.display import Image

Image(filename='amplitude.png')