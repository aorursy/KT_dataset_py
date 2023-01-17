%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.plot(np.linspace(0,50,100), np.sin(np.linspace(0,50,100)))
plt.savefig('out1.png')
from IPython.display import Image, display
display(Image(filename='out1.png', embed=True))
