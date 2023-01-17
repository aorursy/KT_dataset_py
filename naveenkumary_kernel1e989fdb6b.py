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
data = pd.read_csv("../input/insurance/insurance.csv")

data.head()
data.describe()
data.info()
import seaborn as sns

sns.set()
sns.distplot(data["children"])
sns.scatterplot(x="age",y="charges",data=data)
sns.scatterplot(x="children",y="charges",data=data)
sns.scatterplot(x="bmi",y="charges",data=data,alpha=0.5)
log_charges = np.log(data["charges"])

data["log_charges"] = log_charges

data.head()
sns.scatterplot(x="age",y="log_charges",data=data,alpha=0.5)
data_d=pd.get_dummies(data,drop_first=True)

data_d.head()
data_d.columns.values
cols=['log_charges','age', 'bmi', 'children', 'charges', 'sex_male',

       'smoker_yes', 'region_northwest', 'region_southeast',

       'region_southwest']
data_r=data_d[cols]

data_r.head()
data_r=data_r.drop(["charges"],axis=1)

data_r.head()
target = data_r["log_charges"]

inputs = data_r.drop(["log_charges"],axis=1)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(inputs_scaled,target,test_size=0.2,random_state=42)
reg= LinearRegression()

reg.fit(x_train,y_train)
y_hat = reg.predict(x_train)
import matplotlib.pyplot as plt
plt.scatter(y_train, y_hat,alpha=0.3)

plt.xlabel('Targets (y_train)')

plt.ylabel('Predictions (y_hat)')

plt.show()
sns.distplot(y_train-y_hat)
reg_summary = pd.DataFrame(inputs.columns.values,columns=["Features"])
reg_summary["weights"]=reg.coef_

reg_summary
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test, alpha=0.5)

plt.xlabel('Targets (y_test)',size=18)

plt.ylabel('Predictions (y_hat_test)',size=18)

plt.show()
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])

df_pf.head()
y_test=y_test.reset_index(drop=True)
df_pf['Target'] = np.exp(y_test)

df_pf
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']

df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)

df_pf
df_pf.describe()
pd.options.display.max_rows = 999

pd.set_option('display.float_format', lambda x: '%.2f' % x)

df_pf.sort_values(by=['Difference%'])