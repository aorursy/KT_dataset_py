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

d = {'tax' : pd.Series([9.00,9.00,9.00,9.00,8.00,10.00,8.00,8.00,8.00,7.00,8.00,

                        8.00,7.00,7.00,7.00,7.00,7.00,7.00,7.00,7.00,7.00,8.00,9.00,

                        9.00,9.00,9.00,8.00,8.00,8.00,9.00,7.00,7.00,8.00,8.00,8.00,

                        8.00,5.00,5.00,5.00,7.00,7.00,7.00,7.00,7.00,6.00,9.00,7.00,7.00]), 

      'income' : pd.Series([3571,4092,3865,4870,4399,5342,5319,5126,4447,4512,4391,5126,4817,

                         4207,4332,4318,4206,3718,4716,4341,4593,4983,4897,4258,4574,3721,

                         3448,3846,4188,3601,3640,3333,3063,3357,3528,3802,4045,3897,3635,

                         4345,4449,3656,4300,3745,5215,4476,4296,5002]),

      'Highways':pd.Series([1976,1250,1586,2351,431,1333,11868,2138,8577,8507,5939,14186,6930,6580,

                    8159,10340,8508,4725,5915,6010,7834,602,2449,4686,2619,4746,5399,9061,5975

                    ,4650,6905,6594,6524,4121,3495,7834,17782,6385,3274,3905,4639,3985,3635,

                    2611,2302,3942,4083,9794]),

     'driver':pd.Series([0.5250,0.5720,0.5800,0.5290,0.5440,0.5710,0.4510,0.5530,0.5290,0.5520,

                   0.5300,0.5250,0.5740,0.5450,0.6080,0.5860,0.5720,0.5400,0.7240,0.6770,

                   0.6630,0.6020,0.5110,0.5170,0.5510,0.5440,0.5480,0.5790,0.5630,0.4930,

                   0.5180,0.5130,0.5780,0.5470,0.4870,0.6290,0.5660,0.5860,0.6630,0.6720,

                   0.6260,0.5630,0.6030,0.5080,0.6720,0.5710,0.6230,0.5930]),

    'Consumption':pd.Series([541,524,561,414,410,457,344,467,464,498,580,471,525,508,566,635,603,714,

                  865,640,649,540,464,547,460,566,577,631,574,534,571,554,577,628,487,644,

                  640,704,648,968,587,699,632,591,782,510,610,524])}
df = pd.DataFrame(d) 
df.head()
df.info()
df.describe().T['std']
df.isnull().sum()

## There is no missing value
df.corr()
df.corr()['Consumption']

## There is a negative correlation between Petrol Consumption and Petrol Tax
import seaborn as sns

sns.pairplot(df,kind="reg")
sns.jointplot(x="Consumption",y="income",data=df,kind="reg")
sns.jointplot(x="Consumption",y="tax",data=df,kind="reg")
## Linear Regression alternative 1

import statsmodels.api as sm

X=df[["tax"]]

X[0:5]

X=sm.add_constant(X)

X[0:5]

y=df["Consumption"]

lm=sm.OLS(y,X)

model=lm.fit()

model.summary()
## Linear regression alternative 2

import statsmodels.formula.api as smf

lm=smf.ols("Consumption ~ tax",df)

model=lm.fit()

model.summary()
model.params
print(model.summary().tables[1])
## Confidence Interval for data's parameters

model.conf_int()
## Model's f p-value

model.f_pvalue
print("f_pvalue: ","%.4f"%model.f_pvalue)
print("fvalue: ","%.2f"%model.fvalue)
## T value for intercept

print("tvalue: ","%.2f"%model.tvalues[0:1])
## T value for first paramater Petrol Tax

print("tvalue: ","%.2f"%model.tvalues[1:2])
## R squared score - representing score 

model.rsquared_adj
for i in range(10):

    est=(984.6084+(-53.4869*i))

    print("{}".format(i) +". estimation: "+str(est))
## Linear Regression Model

print("Consumption of petrol = "+str("%.2f"%model.params[0])+ " + Petrol Tax "+ "* "+ str("%.2f"%model.params[1]))

## Addition to all of these, you can see the details in the plot attached below. 

## There is a negative regression between Consumption of petrol and Petrol Tax...





g = sns.regplot(df["Consumption"], df["tax"], ci=None, scatter_kws={'color':'r', 's':9})

g.set_title("Model Equation: Consumption of petrol = 984.61 + Petrol Tax*-53.49")

g.set_ylabel("Petrol Tax")

g.set_xlabel("Consumption of petrol")