# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from warnings import filterwarnings

filterwarnings('ignore')

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
ad=pd.read_csv("/kaggle/input/advertisingcsv/Advertising.csv",usecols=[1,2,3,4])

df=ad.copy()

df.head()
#df.iloc[:,1:len(df)]

# df.head()
df.info()
df.describe().T
df.isnull().values.any()
df.corr()
sns.pairplot(df,kind="reg");

sns.jointplot(x="TV",y="Sales",data=df,kind="reg");
import statsmodels.api as sm
X=df[["TV"]]

X[0:5]
X=sm.add_constant(X)

X[0:5]
y=df[["Sales"]]

y[0:5]
lm=sm.OLS(y,X)

model=lm.fit()

model.summary()
# 2nd way

import statsmodels.formula.api as smf

lm=smf.ols("Sales ~ TV",df)

model=lm.fit()

model.summary()
model.params
model.summary().tables[1]
model.conf_int()
model.f_pvalue
print("p value: %.4f"%model.f_pvalue)
print("f value: %.4f"%model.fvalue)
print("t value: %.4f"%model.tvalues[0:1])
model.rsquared_adj
model.fittedvalues[0:5]
y[0:5]
# Model equation asked in data science interviews

print("Sales= %.4f"%model.params[0]+" + TV*%.4f"%model.params[1])
g=sns.regplot("TV","Sales",data=df,ci=None,scatter_kws={'color':'red','s':9})

g.set_title("Model equation : Sales= 7.0326 + TV * 0.0475")

g.set_xlabel("TV")

g.set_ylabel("Sales")

import matplotlib.pyplot as plt

plt.xlim(-10,310)

plt.ylim(0,);
from sklearn.linear_model import LinearRegression

X=df[["TV"]]

y=df["Sales"]

reg=LinearRegression()

model=reg.fit(X,y)

model.intercept_

model.coef_

model.score(X,y)
model.predict(X)[0:10]
7.0326 + 30 * 0.0475
X=df[["TV"]]

y=df["Sales"]

reg=LinearRegression()

model=reg.fit(X,y)
model.predict([[30]])
new_values=[[5],[90],[200]]
model.predict(new_values)
k_t=pd.DataFrame({"real_y":y[0:10],

                  "predict_y":reg.predict(X)[0:10]})

k_t