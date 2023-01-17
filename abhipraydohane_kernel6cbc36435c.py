import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
### The code written beolw is a tutorial for numerical arrays
import numpy as np 



a = np.zeros([3,5])



print(a)



print(np.linspace(0,1,5, endpoint=True, retstep=False))
### The code written below shows how the graphing/plotting tools can be used in Python3.
a = np.zeros(5)

b = np.zeros(5)

for i in range(5):

    a[i] = 2*i

    b[i] = i





plt.plot(b,a)               ### This command plots a vs b  (x,y)



plt.ylabel('The Y axis title')

plt.xlabel('The X axis title')



plt.show()                  ### This command outputs the plot