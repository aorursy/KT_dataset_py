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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

df = pd.read_csv('/kaggle/input/cardiogoodfitness/CardioGoodFitness.csv')
df.head()
df. describe().T
df.hist(figsize = (30, 40)) 

Gender = df['Gender'] 
Age = df['Age']

sns.boxplot(Gender, Age, data = df, )
pd.crosstab(df['Product'], df['Gender'])

sns.countplot(x = 'Product', hue = 'Gender' , data = df)
pd.pivot_table(df, index = ["Gender", "Product"],
                           columns = ["MaritalStatus"], aggfunc = len )


age = df["Age"]
mean = age.mean()

age.std()


age.median()


sns.distplot(age)
r = df.corr()
r

sns.heatmap(r, annot=True)
#Train LinearRegression that predict who is more likely to buy products.
#From previous analysis we assume gender difference is small so we won't consider that.
#Take out income and age and find who is our target costmer.

income = df["Income"]
sns.distplot(income)



mean = income.mean()
mean
income.median()
income.mode()
std = income.std()
std

# No significance difference between mean and median.
# So we will use mean and std to decide our target consumer. 
#Let's say our target costmer's income will be between 53719 +- 16506
import math
Range = ('MAX:',mean + std, "Min:" ,mean - std)
Range 
age = df["Age"]
sns.distplot(age)

age_mean = age.mean()
age_mean

age.median()
age_std = age.std()
age_std
#Same thing again, we will take out mean and caculate +- std and that is going to be our target costmer.

Range = ("Max", age_mean + age_std, "Min", age_mean -  age_std)
Range
# correlation is not strong but not week too.
# Outliers probaly mess around our data. 

r = np.corrcoef(age,income)
r

#Vsidualize data 
#Seems like some outliers mess around with our data. 
from pylab import *

scatter(age, income)
#Some people's income is too high. Remove them from our data.
def reject_outliers(income):
    u = np.median(income)
    s = np.std(income)
    filtered = [e for e in income if (u - 2 * s < e < u + s)]
    return filtered

filtered_income_std1 = reject_outliers(income)

plt.hist(filtered_income_std1, 50)
plt.show()

def reject_outliers(income):
    u = np.median(income)
    s = np.std(income)
    filtered = [e for e in income if (u - 2 * s < e < u + 2 * s)]
    return filtered

filtered_income_std2 = reject_outliers(income)

plt.hist(filtered_income_std2, 50)
plt.show()




#Find corr with sorted income data.
#Pick randmaized 153 data from age data.
import random as rand

sam_age1 = rand.sample(list(age), 153)
sam_age2 = rand.sample(list(age), 163)





r_filtered_income_std1 = np.corrcoef(filtered_income_std1, sam_age1)
r_filtered_income_std1
scatter(filtered_income_std1,sam_age1)
r_filtered_income_std2 = np.corrcoef(filtered_income_std2, sam_age2)
r_filtered_income_std2

#When I am picking up data randomly from age it seems destroying the trande of data that we could see before. 
# Right now I am not sure how to pick up age values to find correlation with sorted income value(That has excluded outliers).


#Find Linear regression bet income and age.
from scipy import stats

slope, intercept, r_value, p_value, std_err = stats.linregress(age, income)


r_value ** 0.5
np.random.seed(2)

scatter(age, income)
X = df['Age'].values
y = df["Income"].values
X = np.reshape(X, (1, -1))
y = np.reshape(y, (1,-1))

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y) 

def viz_linear():
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg.predict(X), color='blue')
    plt.title('Age and income')
    plt.xlabel('Age ')
    plt.ylabel('Income')
    plt.show()
    return
viz_linear()


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Age and income(Linear Regression)')
    plt.xlabel('Age')
    plt.ylabel('Income')
    plt.show()
    return
viz_polymonial()

