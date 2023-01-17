import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import random

x = []

for i in range(0,40):

    x.append(random.uniform(-1,1)) #X in range -1 to 1

    

s = np.random.normal(0,0.1,40) # Normally distributed E
df = pd.DataFrame(data = x, columns = ["Xi"])

df["E"] = s
df["Yi"] = 2*df["Xi"]+df["E"]
df.head()
bRTO = sum(df["Xi"]*df["Yi"]) / sum(df["Xi"]**2)

bRTO
df["YiOrigin"] = (bRTO*df["Xi"])
df.head()
x_mean = df["Xi"].mean()

y_mean = df["Yi"].mean()
sum((df["Xi"] - x_mean)**2)
b1 = sum((df["Xi"] - x_mean)*(df["Yi"] - y_mean)) / sum((df["Xi"] - x_mean)**2) # Value of regression coefficient(Slope)

b1
b0 = y_mean - (b1 * x_mean) # Value of intercept

b0
df["YiHat"] = b0 + (b1*df["Xi"])
df.head()
df["Ei Origin"] = df["Yi"] - df["YiOrigin"]

print("Value of ei for regression through origin =",sum(df["Ei Origin"]))
df["Ei Hat"] = df["Yi"] - df["YiHat"]

print("Value of ei for ordinary linear regression =",sum(df["Ei Hat"]))
df.head()
n = len(df["Xi"])

import math

r_squared_origin = ((n*sum(df["Xi"]*df["YiOrigin"])) - (sum(df["Xi"]) * sum(df["YiOrigin"]))) / math.sqrt((n*sum(df["Xi"]**2))*((n*sum(df["YiOrigin"]**2) - ((sum(df["YiOrigin"]))**2))))
r_squared_origin
r_squared_hat = ((n*sum(df["Xi"]*df["YiHat"])) - (sum(df["Xi"]) * sum(df["YiHat"]))) / math.sqrt((n*sum(df["Xi"]**2))*((n*sum(df["YiHat"]**2) - ((sum(df["YiHat"]))**2))))

r_squared_hat
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

data = pd.read_fwf("/kaggle/input/skincancer2.txt")
data.head()
# y = Mort.    x = lat

lat_mean = sum(data.Lat) / len(data.Lat)

mort_mean = sum(data.Mort) / len(data.Mort)

lat_mean , mort_mean
b1_skn = sum((data.Lat - lat_mean)*(data.Mort - mort_mean)) / sum((data.Lat - lat_mean)**2)

b0_skn = mort_mean - (b1_skn * lat_mean)

print ("Slope =", b1_skn,"Intercept =",b0_skn)
data["predictions"] = (data.Lat * b1_skn) + (b0_skn)

data.head()

mort_mean
SSE = sum((data.Mort - data.predictions)**2)

SST = sum((data.Mort - mort_mean)**2)

SSR = sum((data.predictions - mort_mean)**2)



print ("Sum of Squared Errors = ",SSE," \nSum of Squared due to Regression =",SSR , "\nSum of Squared Total = ",SST) 
import math

MSE = math.sqrt(SSE/len(data.Lat))

yyy = MSE/math.sqrt(sum((data.Lat-lat_mean)**2))
#TO calculate T value

from scipy import stats

ci = 95

n = 48

t = stats.t.ppf(1- ((100-ci)/2/100), n-2)

x = t*yyy
print("The 95% confidence interval turns out to be\n",round(b1_skn - x,3),"< B1 <",round(b1_skn + x,3))
from sklearn.linear_model import LinearRegression

reg = LinearRegression()



train_x = data.iloc[:,1:2]

train_y = data.iloc[:,2:3]

reg.fit(train_x,train_y)



print ("Slope = ",reg.coef_,"\nIntercept = ",reg.intercept_)
from statsmodels.formula.api import ols

model = ols("train_y ~ train_x", data).fit()

model.summary()
import seaborn as sns

sns.regplot(data.Lat,data.Mort)