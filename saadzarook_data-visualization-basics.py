import pandas as pd

import matplotlib.pyplot as plt
#Let's do a very simple example before working with a dataset



x = [1,2,3,4,5]

y = [1,4,9,16,25]

plt.plot(x,y)

plt.title('Simple Example')

plt.xlabel('X-axis')

plt.ylabel('Y-axis')

plt.show()
z = [1,3,9,27,81]

plt.plot(x,y)

plt.plot(x,z)

plt.title('Multiple lines Example')

plt.xlabel('X-axis')

plt.ylabel('Y/Z-axis')

plt.legend(['Y line', 'Z line'])

plt.show()
realEstTrans = pd.read_csv('../input/Sacramentorealestatetransactions.csv')



realEstTrans.head()
plt.plot(realEstTrans.beds, realEstTrans.baths, 'o')

#plt.scatter(realEstTrans.beds, realEstTrans.baths)

plt.show()