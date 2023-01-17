import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

import math
df = pd.read_csv("../input/insurance/insurance.csv")
df.head()
df.info()
unique_vals = df['sex'].value_counts()

print(unique_vals)
df.charges.hist(bins=120)
fig, axes = plt.subplots(1, 2, sharey=True) #We will plot bmi, age



plot_bmi = sns.scatterplot(y = 'charges', x = 'bmi', data=df, ax = axes[0])

plot_age = sns.scatterplot(y = 'charges', x = 'age', data=df, ax = axes[1])



plt.show()
fig, axes = plt.subplots(1,4,sharey=True) #We will plot children, smoker, region, sex; 



plot_children = sns.boxplot(y= 'charges', x="children", data=df,  orient='v' , ax=axes[0])

plot_smoker = sns.boxplot(y= 'charges',x="smoker", data=df,  orient='v' , ax=axes[1])

plot_region = sns.boxplot(y= 'charges',x="region", data=df,  orient='v' , ax=axes[2])

plot_sex = sns.boxplot(y= 'charges',x="sex", data=df,  orient='v' , ax=axes[3])

plot_region.set_xticklabels(labels=df['region'].unique(),rotation = 90)

plot_sex.set_xticklabels(labels=df['sex'].unique(),rotation = 90)

for i in axes[1:4]:

    i.set_ylabel('')   

    

plt.show()
#We have to make use of a function to take out sd and mean for each group to make the code DRY  

def mean_and_sd(indep_var):

    val1 = df[indep_var].unique()[0]

    val2 = df[indep_var].unique()[1]

    var1 = df[df[indep_var] == val1]['charges']

    mean1 = round(var1.mean(),2)

    var2 = df[df[indep_var] == val2]['charges']

    mean2 = round(var2.mean(),2)

    print('{} mean - {} mean = {}'.format(val1,val2,mean1 - mean2))

    #Standard Deviation

    sd1 = np.std(df[df[indep_var] == val1]['charges'])

    sd2 = np.std(df[df[indep_var] == val2]['charges'])

    print('sd of {} is {}, and of {} is {}'.format(val1,round(sd1),val2,round(sd2)))
mean_and_sd('sex')
mean_and_sd('smoker')
from sklearn.preprocessing import OneHotEncoder

onehotencoder = OneHotEncoder(categories='auto')



var1 = onehotencoder.fit_transform(df.region.values.reshape(-1,1)).toarray()

var1 = pd.DataFrame(var1)

var1.columns = ['region_1', 'region_2', 'region_3', 'region_4']

var1 = var1.iloc[:,0:3]

df = pd.concat([df, var1], axis=1)







onehotencoder = OneHotEncoder(categories='auto')

var3 = onehotencoder.fit_transform(df.smoker.values.reshape(-1,1)).toarray()

var3 = pd.DataFrame(var3)

var3.columns = ['smoker_1', 'smoker_2']

var3 = var3.iloc[:,0]

df = pd.concat([df, var3], axis=1)

df = df.drop(columns = ['region','sex','smoker'])
df.head()
df = df[['age', 'bmi', 'children', 'region_1', 'region_2', 'region_3',

       'smoker_1', 'charges']]

df
from sklearn.model_selection import train_test_split

X = df.iloc[:,0:7]

Y = df.iloc[:,7]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 5)
print((x_train.shape,x_test.shape,y_train.shape,y_test.shape))
y_train = np.array(y_train).reshape(-1, 1)

y_train = pd.DataFrame(y_train)

y_test = np.array(y_test).reshape(-1, 1)

y_test = pd.DataFrame(y_test)
print((x_train.shape,x_test.shape,y_train.shape,y_test.shape))
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x_train, y_train)

lr_pred = reg.predict(x_test)

print(reg.coef_)
from sklearn.metrics import mean_squared_error

mean_squared_error(lr_pred,y_test)
fig, ax = plt.subplots()

ax.plot([0,1],[0,1], transform=ax.transAxes)



plt.scatter(lr_pred, y_test)

plt.xlabel("Predicted Values")

plt.ylabel("Observed Values")



plt.show()
r2_lr = r2_score(y_test, lr_pred)

mae_lr = mean_absolute_error(y_test, lr_pred)

mse_lr = mean_squared_error(y_test, lr_pred)

print([r2_lr, mae_lr, mse_lr])
from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor(200)

forest.fit(x_train, y_train)
forest_pred = forest.predict(x_test)
fig, ax = plt.subplots()

ax.plot([0,1],[0,1], transform=ax.transAxes)



plt.scatter(forest_pred, y_test)

plt.xlabel("Predicted Values")

plt.ylabel("Observed Values")



plt.show()
r2_forest = r2_score(y_test, forest_pred)

mae_forest = mean_absolute_error(y_test, forest_pred)

mse_forest = mean_squared_error(y_test, forest_pred)

print([r2_forest, mae_forest, mse_forest])
import xgboost as xgb
xgb_model = xgb.XGBRegressor(objective="reg:linear", random_state=42)



xgb_model.fit(x_train, y_train)

xgb_pred = xgb_model.predict(x_test)
fig, ax = plt.subplots()

ax.plot([0,1],[0,1], transform=ax.transAxes)



plt.scatter(xgb_pred, y_test)

plt.xlabel("Predicted Values")

plt.ylabel("Observed Values")



plt.show()
r2_xgb = r2_score(y_test, xgb_pred)

mae_xgb = mean_absolute_error(y_test, xgb_pred)

mse_xgb = mean_squared_error(y_test, xgb_pred)

print([r2_xgb, mae_xgb, mse_xgb])