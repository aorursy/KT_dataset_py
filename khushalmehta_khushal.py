import matplotlib.pyplot as plt

paraX = range(-50,50)
paraY = [x*x for x in paraX]

plt.plot(paraX,paraY)
plt.show()
'''import matplotlib.pyplot as plt

def parabola(x):
    return(x*x)

parax = []
paray = []

for i in range(-1000,1000):
    parax.append(random.randint(-1000,1000))
    
for i in range(0,len(parax)):
    paray.append(parabola(parax[i]))

plt.plot(parax,paray)
plt.show()



import matplotlib.pyplot as plt

def hyperbola(a,b,x):
    return((b/a)*((x*x-a*a)**0.5))

hyperx = []
hypery = []

for i in range(-1000,1000):
    hyperx.append(random.randint(-1000,1000))
    
for i in range(0,len(parax)):
    hypery.append(hyperbola(2,3,hyperx[i]))

plt.plot(hyperx,hypery)
plt.show()



import matplotlib.pyplot as plt
import numpy as np
def cosine(x):
    return(np.cos(x))

parax = []
paray = []

for i in range(0,2*np.pi):
    parax.append(random.randint(0,2*np.pi))
    
for i in range(0,2*np.pi):
    paray.append(cosine(parax[i]))

plt.plot(parax,paray)
plt.show()



import numpy as np 
import matplotlib.pyplot as plt 
  
x = np.linspace(-(2*np.pi), 2*np.pi,20) 
y = np.cos(x) 
  
plt.plot(x,y)
plt.show() '''

import numpy as np 
import matplotlib.pyplot as plt 
  
x = np.linspace(-(2*np.pi), 2*np.pi,20) 
y = np.sin(x) 
  
plt.plot(x,y)
plt.show() 






