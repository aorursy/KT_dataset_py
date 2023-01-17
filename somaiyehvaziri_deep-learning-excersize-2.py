# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
myData = np.array([[1,3,5,7], [2,4,6,8], [0,10,11,12]])  # Defining a matrix with 3 rows and 4 columns
print(myData.shape)         # Prints (3, 4)
myData2 = myData[:2, 1:3] # Creates a subarray consisting of the first 2 rows, and columns 1 and 2
# [[35]
#  [46]]

print(myData[0, 1])     # Prints "3"
myData2[0, 0] = 20                                    # myData2[0, 0] is the same piece of data as myData[0, 1]
print(myData[0, 1])                                    # Prints "20"

row1 = myData[2, :]                                   # Rank 1 view of the third row of myData
row2 = myData[1:2, :]   # Rank 2 view of the second row of myData
print(row1, row1.shape)             # Prints "[246 8] (4,)"
print(row2, row2.shape)               # Prints "[[246 8]] (1, 4)"

myData = np.array([0, 2, 4, 6, 8])
print(myData[-1])       # Prints "8"
print(myData[-4])       # Prints "2"
myData= np.array([0, 2, 4, 6, 8])
print(myData[-3:])        # Prints "[4 68]"

myData = np.array([[0, 2, 4],
	    [6, 8, 10],
	    [12, 14, 16]])
# separate data
X, y = myData[:, :-1], myData[:, -1]
print(X)
print(y)
import numpy as np
np.arange(4)            # Prints "([0, 1, 2, 3])"
np.arange(4.0)           # Prints "([ 0.,  1.,  2., 3.])"
np.arange(2,5)           # Prints "([2, 3, 4])"#
np.arange(3, 9, 2)        # Prints "([3, 5, 7])"    # np.arange(start, stop,  step); the interval does not include stop value

myData1 = np.array([[3,4],[5,6]])
myData2 = np.array([[7,8],[0,1]])

print(myData1 + myData2)
print(np.add(myData1, myData2))

print(np.sqrt(myData1))
import numpy as np

myData1 = np.array([[2,4]])
myData2 = np.array([[1],[3]])
myData3 = np.array([[1,3], [2, 5]])
myData4 = np.array([[0,1], [2, 3]])

# Inner product of vectors
print(np.dot(myData1, myData2))                                 # Prints "14"

# Matrix / vector product
print(np.dot(myData1, myData3))                                # Prints "36"

# Matrix / matrix product
print(np.dot(myData3, myData4))
# [[610]
#  [10 17]]









myData = np.array([1, 3, 5, 7, 0, 2, 4, 6])
myData = myData.reshape((2, 4))
print (myData)
myData = np.array([2, 3, 4, 5, 6])
print(myData.shape)
print (myData.shape[0])                 # myData.shape[0] indicates the first dimension shape
# reshape
myData = myData.reshape((myData.shape[0], 1))      # reshapes the array to have 5 rows with 1 column
print(myData.shape)
print (myData)
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
import pandas as pd
url = '../input/pima-indians-diabetes-database/diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(url, names=names)
print(myData.shape)


import pandas as pd
myFilename = '../input/pima-indians-diabetes-database/diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myFilename)
peek = myData.head(10)
print(peek)
import pandas as pd
myFilename = '../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myFilename, names=names)
types = myData.dtypes
print(types)

import pandas as pd
myFilename = '../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myFilename, names=names)
outcome_counts = myData.groupby('Outcome').size()
print(outcome_counts)
import pandas as pd
from pandas import set_option
myFilename = '../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myFilename, names=names)
set_option('display.width', 200)
set_option('precision', 3)
correlations = myData.corr(method='pearson')
print(correlations)

import matplotlib.pyplot as plt
import pandas as pd
myFilename = '../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myFilename, names=names)
myData.plot.hist()
myData.hist()

import matplotlib.pyplot as plt
import pandas as pd
myFilename = '../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myFilename, names=names)
myData.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
myFilename = '../input/pimaindiansdiabetescsv/pima-indians-diabetes.csv'
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
myData = pd.read_csv(myFilename, names=names)
correlations = myData.corr()
# plot correlation matrix
myfig = plt.figure()
axis = myfig.add_subplot(111)      # There is only one subplot or graph; 
# "111" means "1x1 grid, first subplot"
cax = axis.matshow(correlations, vmin=-1, vmax=1)
myfig.colorbar(cax)
ticks = np.arange(0,9,1)   # np.arange(start, stop,  step); the interval does not include stop value
axis.set_xticks(ticks)
axis.set_yticks(ticks)
axis.set_xticklabels(names)
axis.set_yticklabels(names)
plt.show()