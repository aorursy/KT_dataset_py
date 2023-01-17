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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
df = pd.read_csv("/kaggle/input/red-wine-quality-cortez-et-al-2009/winequality-red.csv")
df.head()
df.info()
df.describe().T
df['quality'].value_counts()
fig = plt.figure(figsize = (10,6))
sns.pairplot(df)
fig = plt.figure(figsize = (10,6))
sns.heatmap(df.corr(),annot=True)

X = df.drop('quality',axis=1)
y = df['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 42)
import statsmodels.api as sm
X_train_new = sm.add_constant(X_train)
X_test_new = sm.add_constant(X_test)
full_model = sm.OLS(y_train,X_train_new)
full_res = full_model.fit()
full_res.summary()
print("Variance inflation Factor")
cnames = X_train.columns
for i in np.arange(0,len(cnames)):
    xvars = list(cnames)
    yvars = xvars.pop(i)
    mod = sm.OLS(X_train[yvars],sm.add_constant((X_train_new[xvars])))
    res= mod.fit()
    vif=1/(1-res.rsquared)
    print(yvars,round(vif,3))
from statsmodels.nonparametric.smoothers_lowess import lowess
residuals = full_res.resid
fitted = full_res.fittedvalues

smoothed = lowess(residuals,fitted)
top3 = abs(residuals).sort_values(ascending = False)[:3]

plt.rcParams.update({'font.size': 16})
plt.rcParams["figure.figsize"] = (8,7)
fig, ax = plt.subplots()
ax.scatter(fitted, residuals, edgecolors = 'k', facecolors = 'none')
ax.plot(smoothed[:,0],smoothed[:,1],color = 'r')
ax.set_ylabel('Residuals')
ax.set_xlabel('Fitted Values')
ax.set_title('Residuals vs. Fitted')
ax.plot([min(fitted),max(fitted)],[0,0],color = 'k',linestyle = ':', alpha = .3)

for i in top3.index:
    ax.annotate(i,xy=(fitted[i],residuals[i]))

plt.show()
