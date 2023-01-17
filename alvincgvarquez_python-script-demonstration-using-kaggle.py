import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt



dx = 0.1

x = np.arange(0.,100+dx,dx)

#print(x)



y = x**2.



plt.plot(x,y)

plt.show()
import numpy as np

import matplotlib.pyplot as plt



dx = 0.1

x = np.arange(0.,100+dx,dx)

#print(x)



y = x**2.



plt.plot(x,y,color='black',linestyle='-.')

plt.title("Title")

plt.xlabel("Horizontal axis label")

plt.ylabel("Vertical axis label")

plt.show()