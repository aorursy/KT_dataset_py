import numpy as np

import matplotlib.pyplot as plt



x=np.arange(0,10)

y=x^2



#Plot graph

plt.plot(x,y)

import numpy as np

import matplotlib.pyplot as plt



x=np.arange(0,10)

y=x^2



#Labling the axes and title

plt.title("Graph Drawing")

plt.xlabel("Time")

plt.ylabel("Distance")



#Plot graph

plt.plot(x,y)
import numpy as np

import matplotlib.pyplot as plt



x=np.arange(0,10)

y=x^2



#Labling the axes and title

plt.title("Graph Drawing")

plt.xlabel("Time")

plt.ylabel("Distance")



#formating the line colour

plt.plot(x,y,"g")



#formating the line type

plt.plot(x,y,">")

import numpy as np

import matplotlib.pyplot as plt



x=np.arange(0,10)

y=x^2



#Labling the axes and title

plt.title("Graph Drawing")

plt.xlabel("Time")

plt.ylabel("Distance")



#formating the line colour

plt.plot(x,y,"g")



#formating the line type

plt.plot(x,y,">")



#save in pdf format

plt.savefig("timevsdist.pdf",format="pdf")