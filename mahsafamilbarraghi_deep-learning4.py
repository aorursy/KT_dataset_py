import matplotlib.pyplot as plt
import numpy as np
myData = np.array([3, 5, 7])
plt.plot(myData)
plt.xlabel('data specification for x axis')
plt.ylabel('data specification for y axis')
plt.show()

import matplotlib.pyplot as plt
import numpy as np
myData1 = np.array([2, 4, 6])
myData2 = np.array([1, 3, 5])
plt.scatter(myData1, myData2)
plt.xlabel('data specification for x axis')
plt.ylabel('data specification for y axis')
plt.show()

import numpy as np
import pandas as pd
myData = np.array([[0, 2, 4], [1, 3, 5]])
row_names = ['row 1', 'row 2']
col_names = ['first', 'second', 'third']
dataframe = pd.DataFrame(myData, index=row_names, columns=col_names)
print(dataframe)
