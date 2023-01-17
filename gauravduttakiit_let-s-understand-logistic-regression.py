import pandas as pd

import numpy as np

import seaborn as sns
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dib = pd.read_csv('/kaggle/input/Diabetes Example Data.csv')
dib.head()
# Converting Yes to 1 and No to 0

dib['Diabetes'] = dib['Diabetes'].map({'Yes': 1, 'No': 0})
dib.head()
dib.shape
dib.describe()
sns.jointplot(x = dib['Blood Sugar Level'],y = dib['Diabetes'],kind="kde", color="r");
dib['Diabetes'].value_counts()
sns.countplot(x='Diabetes',data=dib);
sns.heatmap(dib.corr(), annot = True, cmap="tab20c");
# Putting feature variable to X

X = dib['Blood Sugar Level']



# Putting response variable to y

y = dib['Diabetes']
import statsmodels.api as sm
logm1 = sm.GLM(y,(sm.add_constant(X)), family = sm.families.Binomial())

result=logm1.fit().summary()

result