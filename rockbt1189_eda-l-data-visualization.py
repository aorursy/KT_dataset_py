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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
df = pd.read_csv("../input/insurance.csv")

df.head()
print(df.shape)

print(df.describe())

print("Total NO. of null value in the dataset: ",df.isnull().sum())
#As per BMI group the rick type i.e. Normal<=24,24.1 to 29.9 Overweight,30 or more Obese

df["risk_type"] = np.where(df.bmi<=24,"Normal",(np.where(df.bmi  <30,"OverWeight","obese")))

df["Age_Group"] = np.where(df.age<25,"Age below 25 year",(np.where(df.age<35, "Age 25 to 34 year" ,

                                                                   (np.where(df.age<55, "Age 35 to 54 year" ,

                                                                             (np.where(df.age<75, "Age 55 to 74 year" ,"Age more then 75 year")))))))



df.head()
df["Charge"]=df["charges"]/1000
#Correlation:

sns.pairplot(df,height=1.8)
#

plt.rcParams["figure.figsize"] = (20,14)

plt.subplot(421)

df['age'].value_counts().sort_index().plot.line(color="k")

plt.title("Age distribution in the Data")



plt.subplot(422)

df['bmi'].sort_index().plot.hist(color="g")

plt.title("bmi distribution in the Data")



plt.subplot(423)

df['children'].value_counts().plot.line(color="b")

plt.title("child distribution in the Data")



plt.subplot(424)

df['charges'].plot.hist(color="c")

plt.title("charges distribution in the Data")





plt.subplot(425)

df["risk_type"].value_counts().plot.bar()

plt.title("risk_type distribution in the data")



plt.subplot(426)

df["smoker"].value_counts().plot.bar()

plt.title("smoker distribution in the data")



plt.subplot(427)

df["Age_Group"].value_counts().plot.bar()

plt.title("Age_Group distribution in the data")
# See Age Vs BMI Vs Charges



plt.rcParams["figure.figsize"]=(18,8)

plt.subplot(221)

sns.lineplot(x="age",y="bmi",data=df,color="b")



plt.subplot(222)

sns.lineplot(x="age",y="Charge",data=df,color="g")



plt.subplot(223)

sns.scatterplot(x="Charge",y="bmi",data=df,color="k")



#Age_Group wise BMI and Charges

plt.rcParams["figure.figsize"]=(18,8)

plt.subplot(221)

sns.lineplot(x="age",y="bmi",data=df,hue="Age_Group")



plt.subplot(222)

sns.lineplot(x="age",y="Charge",data=df,hue="Age_Group")
#Smoker wise BMI and Charges

plt.rcParams["figure.figsize"]=(18,8)

plt.subplot(221)

sns.lineplot(x="age",y="bmi",data=df,color="b",hue="smoker")



plt.subplot(222)

sns.lineplot(x="age",y="Charge",data=df,color="g",hue="smoker")
#region wise Age vs BMI and Charges

southwest = df[df["region"]=="southwest"]

southeast = df[df["region"]=="southeast"]

northwest = df[df["region"]=="northwest"]

northeast = df[df["region"]=="northeast"]



plt.rcParams["figure.figsize"]=(18,8)

plt.subplot(421)#region wise Age vs BMI and Charges

southwest = df[df["region"]=="southwest"]

southeast = df[df["region"]=="southeast"]

northwest = df[df["region"]=="northwest"]

northeast = df[df["region"]=="northeast"]



plt.rcParams["figure.figsize"]=(18,8)

plt.subplot(421)

sns.lineplot(x="age",y="bmi",data=southwest,hue="risk_type",hue_order=["Normal","OverWeight","obese"])

plt.title("southwest Region")

plt.subplot(422)

sns.lineplot(x="bmi",y="Charge",data=southwest,hue="risk_type",hue_order=["Normal","OverWeight","obese"])

plt.title("southwest Region")



plt.subplot(423)

sns.lineplot(x="age",y="bmi",data=southeast,hue="risk_type",hue_order=["Normal","OverWeight","obese"])

plt.title("southeast Region")

plt.subplot(424)

sns.lineplot(x="bmi",y="Charge",data=southeast,hue="risk_type",hue_order=["Normal","OverWeight","obese"])

plt.title("southeast Region")



plt.subplot(425)

sns.lineplot(x="age",y="bmi",data=northwest,hue="risk_type",hue_order=["Normal","OverWeight","obese"])

plt.title("northwest Region")

plt.subplot(426)

sns.lineplot(x="bmi",y="Charge",data=northwest,hue="risk_type",hue_order=["Normal","OverWeight","obese"])

plt.title("northwest Region")



plt.subplot(427)

sns.lineplot(x="age",y="bmi",data=northeast,hue="risk_type",hue_order=["Normal","OverWeight","obese"])

plt.title("northeast Region")

plt.subplot(428)

sns.lineplot(x="bmi",y="Charge",data=northeast,hue="risk_type",hue_order=["Normal","OverWeight","obese"])

plt.title("northeast Region")
plt.rcParams["figure.figsize"]=(16,6)

plt.subplot(2,2,1)

sns.violinplot(x="sex",y="bmi",data=df)



plt.subplot(2,2,2)

sns.violinplot(x="smoker",y="bmi",data=df)



plt.subplot(2,2,3)

sns.violinplot(x="risk_type",y="bmi",data=df)



plt.subplot(2,2,4)

sns.violinplot(x="Age_Group",y="bmi",data=df)
plt.rcParams["figure.figsize"]=(16,6)

plt.subplot(2,2,1)

sns.violinplot(x="sex",y="Charge",data=df)



plt.subplot(2,2,2)

sns.violinplot(x="smoker",y="Charge",data=df)



plt.subplot(2,2,3)

sns.violinplot(x="risk_type",y="Charge",data=df)



plt.subplot(2,2,4)

sns.violinplot(x="Age_Group",y="Charge",data=df)
plt.rcParams["figure.figsize"]=(28,10)

plt.subplot(2,2,1)

sns.violinplot(x="children",y="bmi",data=df,hue="smoker")



plt.rcParams["figure.figsize"]=(28,10)

plt.subplot(2,2,3)

sns.violinplot(x="children",y="Charge",data=df,hue="smoker")
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.ensemble import RandomForestRegressor