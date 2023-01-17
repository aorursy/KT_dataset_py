import numpy as np

import pandas as pd

import matplotlib.pyplot as plt, seaborn as sns
am0=pd.read_csv("/kaggle/input/AutoData.csv")
am0.head()
am0.describe()
am0.describe(include=['object'])
am0.info()
am0.isnull().sum()
am0.describe(include=['object'])
am0.price.plot.box()

plt.show()
am0.make.value_counts(normalize=True)
am0.fueltype.value_counts(normalize=True)
am0.carbody.value_counts(normalize=True)
am0.aspiration.value_counts(normalize=True)
am0.doornumber.value_counts(normalize=True)
am0.drivewheel.value_counts(normalize=True)
am0.enginelocation.value_counts(normalize=True)
am0.cylindernumber.value_counts(normalize=True)
am0.fuelsystem.value_counts(normalize=True)
am0.groupby(["fueltype"])['price'].mean()
am0.groupby(["carbody"])['price'].mean()
am0.groupby(["aspiration"])['price'].mean()
am0.groupby(["doornumber"])['price'].mean()
am0.groupby(["drivewheel"])['price'].mean()
am0.groupby(["enginelocation"])['price'].mean()
am0.groupby(["cylindernumber"])['price'].mean()
am0.groupby(["fuelsystem"])['price'].mean()
sns.pairplot(am0, x_vars=['curbweight','enginesize','boreratio','stroke','compressionratio'],y_vars='price', markers="+", diag_kind="kde")

plt.show()
sns.pairplot(am0, x_vars=['symboling','wheelbase','carlength','carwidth','carheight'],y_vars='price', markers="+", diag_kind="kde")

plt.show()
sns.pairplot(am0, x_vars=['horsepower','peakrpm','citympg','highwaympg'],y_vars='price', markers="+", diag_kind="kde")

plt.show()
am1 = am0.copy()
from sklearn.preprocessing import LabelEncoder
# LabelEncoder

le = LabelEncoder()



# apply "le.fit_transform"

am1['symboling']=le.fit_transform(am0['symboling'])

am1['symboling'].unique()
am1['make']=le.fit_transform(am0['make'])

am1['make'].unique()
am1['fueltype']=le.fit_transform(am0['fueltype'])

am1['fueltype'].unique()
am1['aspiration']=le.fit_transform(am0['aspiration'])

am1['aspiration'].unique()
am1['doornumber']=le.fit_transform(am0['doornumber'])

am1['doornumber'].unique()
am1['carbody']=le.fit_transform(am0['carbody'])

am1['carbody'].unique()
am1['drivewheel']=le.fit_transform(am0['drivewheel'])

am1['drivewheel'].unique()
am1['enginelocation']=le.fit_transform(am0['enginelocation'])

am1['enginelocation'].unique()
am1['enginetype']=le.fit_transform(am0['enginetype'])

am1['enginetype'].unique()
am1['cylindernumber']=le.fit_transform(am0['cylindernumber'])

am1['cylindernumber'].unique()
am1['fuelsystem']=le.fit_transform(am0['fuelsystem'])

am1['fuelsystem'].unique()
am1.head()
res= am1.corr()

res
plt.figure(figsize=[18,18])

sns.heatmap(res, cmap="Reds", annot=True)

plt.show()
cor = am0.corr()

cor
sns.jointplot('horsepower', 'price', data = am0, kind="reg")

plt.show()
sns.jointplot('enginesize', 'price', data = am0, kind="reg")

plt.show()
sns.jointplot('curbweight', 'price', data = am0, kind="reg")

plt.show()
X = am1[['enginesize']]

y = am1['price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, random_state = 42)
X_train.shape, X_test.shape
from sklearn.linear_model import LinearRegression
#Instantiating the linear regression model

mod = LinearRegression()
mod.fit(X_train, y_train)
mod.intercept_, mod.coef_
from sklearn.metrics import r2_score
y_train_pred = mod.predict(X_train)
r2_score(y_train, y_train_pred)
X = am1.drop("price", axis=1)

X.head()
y = am1[['price']]

y.head()
df_train, df_test = train_test_split(am1, train_size = 0.7, random_state = 42)
df_train.shape, df_test.shape
y_train = df_train[['price']]

X_train = df_train

y_test = df_test[['price']]

X_test = df_test
from sklearn.linear_model import LinearRegression
mod1= LinearRegression()
mod1.fit(X_train, y_train)
X_train.columns
from sklearn.metrics import r2_score
y_train_pred = mod1.predict(X_train)
r2_score(y_train, y_train_pred)
from sklearn.feature_selection import RFE
mod2 = LinearRegression()
rfe_selector = RFE(mod2, 10)
rfe_selector.fit(X_train, y_train)
rfe_selector.support_
rfe_selector.ranking_
cols_keep = X_train.columns[rfe_selector.support_]
cols_keep
lr2 = LinearRegression()
lr2.fit(X_train[cols_keep],y_train)
y_train_pred = lr2.predict(X_train[cols_keep])
r2_score(y_train, y_train_pred)
import statsmodels.api as sm
X_train.head()
X_train_sm = sm.add_constant(X_train[cols_keep])

X_train_sm.head()
lr = sm.OLS(y_train, X_train_sm).fit()
lr.summary()
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()

vif['Features'] = X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
X_train1_sm = sm.add_constant(X_train)

X_train1_sm.head()
lr1 = sm.OLS(y_train, X_train_sm).fit()
lr1.summary()
y_train = df_train[['price']]

X_train = df_train[['enginesize','curbweight','horsepower']]

y_test = df_test[['price']]

X_test = df_test[['enginesize','curbweight','horsepower']]
#Instantiating the linear regression model

mod3 = LinearRegression()

mod3.fit(X_train, y_train)
y_train_pred = mod3.predict(X_train)

r2_score(y_train, y_train_pred)