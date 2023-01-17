file = '../input/set_a/normal__201104141251.wav'
# first, we need to import our essentia module. It is aptly named 'essentia'!
import essentia

# as there are 2 operating modes in essentia which have the same algorithms,
# these latter are dispatched into 2 submodules:
import essentia.standard
import essentia.streaming

import IPython
# we start by instantiating the audio loader:
loader = essentia.standard.MonoLoader(filename=file)

# and then we actually perform the loading:
audio = loader()
IPython.display.Audio(file)
# pylab contains the plot() function, as well as figure, etc... (same names as Matlab)
from pylab import plot, show, figure, imshow
%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default

plot(audio)
plt.title("Heart beats")
show() # unnecessary if you started "ipython --pylab"