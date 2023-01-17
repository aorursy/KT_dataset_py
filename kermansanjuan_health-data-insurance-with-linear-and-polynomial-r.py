import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as ms

from sklearn.linear_model import LinearRegression
df= pd.read_csv('../input/insurance/insurance.csv')
df.head()
df.isnull().sum()
df['sex'].replace('female',0,inplace = True)

df['sex'].replace('male',1,inplace = True)
df['smoker'].replace('yes',1,inplace = True)

df['smoker'].replace('no',0,inplace = True)
df.head()
df.dtypes
df.describe()
df.corr()['charges'].sort_values()
df.groupby("smoker").mean()
f= plt.figure(figsize=(12,5))

ax=f.add_subplot(121)

sns.distplot(df[(df.smoker == 1)]['age'],color='b',ax=ax)
sns.catplot(x="smoker", kind="count",hue = 'sex', palette="Paired", data=df)
sns.distplot(df[(df.smoker == 0)]["charges"],color='g').set(title = 'Insurance cost comparation between smoking or not')

sns.distplot(df[(df.smoker == 1)]["charges"],color='b').set(title = 'Insurance cost comparation between smoking or not')
df_gptest = df[['sex','smoker','charges','age']]

grouped_test1 = df_gptest.groupby(['sex','smoker'],as_index=False).mean()

grouped_test1
sns.lmplot(x="age", y="charges", hue="smoker", data=df, palette = 'Paired', height = 7)

ax.set_title('Smokers and non-smokers')
width = 12

height = 10

plt.figure(figsize=(width, height))

sns.regplot(x="age", y="charges", data=df)

plt.ylim(0,)
from sklearn.model_selection import train_test_split

y_data = df['charges']

x_data = df.drop('charges',axis=1) #All the data except the one we want to predict it.



x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)





print("number of test samples :", x_test.shape[0])

print("number of training samples:",x_train.shape[0])



ln = LinearRegression()
ln.fit(x_train[['age', 'sex', 'bmi','children','smoker']], y_train)
yhat_train = ln.predict(x_train[['age', 'sex', 'bmi','children','smoker']])
yhat_test = ln.predict(x_test[['age', 'sex', 'bmi','children','smoker']])
def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):

    width = 12

    height = 10

    plt.figure(figsize=(width, height))



    ax1 = sns.distplot(RedFunction, hist=False, color="r", label=RedName)

    ax2 = sns.distplot(BlueFunction, hist=False, color="b", label=BlueName, ax=ax1)



    plt.title(Title)

    plt.xlabel('Charge (in dollars)')

    plt.ylabel('Provided Features')



    plt.show()

    plt.close()
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'

DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

ln.score(x_train[['age', 'sex', 'bmi','children','smoker']], y_train)
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'

DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
ln.score(x_test[['age', 'sex', 'bmi','children','smoker']], y_test) #Not the best score...
from sklearn.preprocessing import PolynomialFeatures
Rsqu_test = []



order = [1, 2, 3, 4]

for n in order:

    pr = PolynomialFeatures(degree=n)

    

    x_train_pr = pr.fit_transform(x_train[['age', 'sex', 'bmi','children','smoker']])

    

    x_test_pr = pr.fit_transform(x_test[['age', 'sex', 'bmi','children','smoker']])    

    

    ln.fit(x_train_pr, y_train)

    

    Rsqu_test.append(ln.score(x_test_pr, y_test))



plt.plot(order, Rsqu_test)

plt.xlabel('order')

plt.ylabel('R^2')

plt.title('R^2 Using Test Data')

plt.text(3, 0.75, 'Maximum R^2 ')    
pr = PolynomialFeatures(degree=3)

x_train_pr = pr.fit_transform(x_train[['age', 'sex', 'bmi','children','smoker']])

x_test_pr = pr.fit_transform(x_test[['age', 'sex', 'bmi','children','smoker']])

poly = LinearRegression()

poly.fit(x_train_pr, y_train)

yhat_train = poly.predict(x_train_pr)

yhat_test = poly.predict(x_test_pr)
Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'

DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)
poly.score(x_train_pr, y_train)
Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'

DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)
poly.score(x_test_pr, y_test)