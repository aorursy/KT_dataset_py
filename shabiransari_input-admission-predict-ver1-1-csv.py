# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
df = pd.read_csv("../input/Admission_Predict_Ver1.1.csv", index_col=0)
print(df.head())

print(df.columns)
print(df.info())
print(df.describe())

import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df)
plt.show()
print(max(df['GRE Score']))
print(min(df['GRE Score']))
import matplotlib.pyplot as plt
Chance_of_Admit=df['Chance of Admit ']
GRE_Score = df['GRE Score']
plt.xlabel('Chance of Admit')
plt.ylabel('GRE Score')
plt.plot(Chance_of_Admit,GRE_Score,marker='D', color='red')
plt.show()
#above fig. is not clear we can costomize it
import matplotlib.pyplot as plt
Chance_of_Admit=df['Chance of Admit ']
GRE_Score = df['GRE Score']
plt.xlabel('Chance of Admit')
plt.ylabel('GRE Score')
plt.hist(Chance_of_Admit,GRE_Score,marker='D', color='red')
plt.xlim(0.9, 1.0)
plt.ylim(330, 340)
plt.show()

import seaborn as sns
sns.jointplot(x='Chance of Admit ', y='GRE Score', data=df)
plt.show()


import seaborn as sns
sns.jointplot(x='Chance of Admit ', y='CGPA', data=df)
plt.show()

sns.swarmplot(x='Chance of Admit ', y='GRE Score', data=df)
plt.show()