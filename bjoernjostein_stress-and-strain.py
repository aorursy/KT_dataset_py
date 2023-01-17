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
os.mkdir("/kaggle/working/val")
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

S_S = pd.read_csv("/kaggle/input/sscurve2/StressAndStrain.csv")
S_S
plt.style.use('seaborn-bright')

print(plt.style.available)

print("NumPy version:", np.__version__)

print("Pandas version:", pd.__version__)

z = np.polyfit(S_S['Strain'][0:5], S_S['Stress'][0:5], 1)
z
z[1] = -1
z
#S_S = pd.read_excel("SSCurve.xls")

print(S_S.head())

Stress_Al = S_S['Stress']

Strain_Al = S_S['Strain']

plt.scatter(Strain_Al, Stress_Al)

plt.plot(Strain_Al, Stress_Al)

plt.xlabel('Strain')

plt.ylabel('Stress (MPa)')

plt.title('Engineering Stress Vs Engineering Strain')

plt.ylim(0,7)

x= Strain_Al

plt.plot(z[0]*x+z[1],'--')

plt.tight_layout()

plt.show()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('seaborn-bright')

print(plt.style.available)

print("NumPy version:", np.__version__)

print("Pandas version:", pd.__version__)

S_S = pd.read_csv("/kaggle/input/sscurve2/StressAndStrain.csv")

print(S_S.head())

Stress_Al = S_S['Stress']

Strain_Al = S_S['Strain']

z = np.polyfit(S_S['Strain'][0:5], S_S['Stress'][0:5], 1)

z[1] = -1

plt.scatter(Strain_Al, Stress_Al)

plt.plot(Strain_Al, Stress_Al)

plt.xlabel('Strain')

plt.ylabel('Stress (MPa)')

plt.title('Engineering Stress Vs Engineering Strain')

plt.ylim(0,7)

x= Strain_Al

plt.plot(z[0]*x+z[1],'--')

plt.tight_layout()

plt.show()