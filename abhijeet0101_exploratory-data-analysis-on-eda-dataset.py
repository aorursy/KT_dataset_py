import seaborn as sns

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
import pandas as pd
z = pd.read_excel("../input/EXDA_4.xlsx")

z.head()
sns.boxplot(data=z,y='Salary')
z.describe()
plt.figure(figsize=(15,6))

sns.countplot(x='Agency',data=z)

plt.tight_layout()
z.groupby('Agency').count()
z.groupby(['Agency','Sub-Agency ID']).count()
z.groupby(['Agency','Sub-Agency ID']).count().reset_index().groupby('Agency').count()
z1=z[(z.Agency=='General Services') & (z.Salary.notna())]

sns.distplot(z1['Salary'])
z2=z[(z.Agency=='Law Department') & (z.Salary.notna())]

sns.distplot(z2['Salary'])