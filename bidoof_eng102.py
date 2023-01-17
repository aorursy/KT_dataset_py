import numpy as np

import numpy.fft as nf

import matplotlib.pyplot as plt
def waveform(n, p, color = "b"):

    t = np.linspace(0, n, n*20)

    plt.xlabel("seconds")

    plt.plot( t,np.sin(2*np.pi*(t/p)), color ) #creates a sine wave with a period of 1 cycle every 50 seconds
color = ""

waveform(50, 50, color)

plt.show()
waveform(100, 50, "g")

waveform(100, 25, "y--")

plt.legend(["freq = 50", "freq = 25"])

plt.show()
def signal(numWaves):

    t = np.linspace(0,100,1000)

    wave = np.zeros(1000)

    for i in range(numWaves):

        wave += np.sin(t*np.random.randint(1,10)/10*(2*np.pi))

    return [t,wave]

        
t,sgnl = signal(3)

plt.plot(t,sgnl)

plt.show()