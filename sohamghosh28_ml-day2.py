# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

emp=pd.read_csv("../input/employee/employee.csv")

plt.hist(emp['Age'],facecolor='blue',bins=5)
import pandas as pd

import matplotlib.pyplot as plt

emp=pd.read_csv("../input/employee/employee.csv")

plt.hist(emp['Sales'],facecolor='orange',bins=10)
import pandas as pd

import matplotlib.pyplot as plt

emp=pd.read_csv("../input/employee/employee.csv")

plt.hist(emp['Profit'],facecolor='green',bins=15)

import pandas as pd

import matplotlib.pyplot as plt

emp=pd.read_csv("../input/employee/employee.csv")

plt.hist(emp['Age'],facecolor='blue',bins=5)

plt.hist(emp['Sales'],facecolor='orange',bins=10)

plt.hist(emp['Profit'],facecolor='green',bins=15)

import matplotlib.pyplot as plt



Age = [24,26,45,32,29,50,34,45,35]

my_labels = 'E001','E002','E004','E008','E010','E011','E025','E035','E020'

plt.pie(Age,labels=my_labels,autopct='%1.1f%%')

plt.title('Age')

plt.axis('equal')

plt.show()
import matplotlib.pyplot as plt



Age = [24,26,45,32,29,50,34,45,35]

my_explode=(0,0,0,0.1,0,0,0,0,0)

my_labels = 'E001','E002','E004','E008','E010','E011','E025','E035','E020'

plt.pie(Age,labels=my_labels,autopct='%1.1f%%',explode=my_explode)

plt.title('Age')

plt.axis('equal')

plt.show()
import pandas as pd

import matplotlib.pyplot as plt

emp=pd.read_csv("../input/employee/employee.csv")

emp.boxplot(by='Age',grid=False)