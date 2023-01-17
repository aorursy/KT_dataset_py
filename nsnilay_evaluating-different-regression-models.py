import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



house_df = pd.read_csv('../input/train.csv')

numeric_data = house_df.select_dtypes(include=['int64','float64'])

for data in numeric_data:

    numeric_data[data] = numeric_data[data].fillna(numeric_data[data].mean())



numeric_data.describe()

del numeric_data['Id']

del numeric_data['YrSold']

del numeric_data['MoSold']

del numeric_data['MiscVal']

del numeric_data['PoolArea']

del numeric_data['ScreenPorch']

del numeric_data['3SsnPorch']

del numeric_data['LowQualFinSF']

del numeric_data['BsmtFinSF2']

numeric_data.describe()
numeric_data_corr = numeric_data.corr()

import seaborn as sns

import matplotlib.pyplot as plt

mask = np.zeros_like(numeric_data_corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

# # Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))

#

# # Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)

#

# # Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(numeric_data_corr, mask=mask, cmap=cmap,vmax=.8,

            square=True,linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
u=np.array(numeric_data)

X=u[:,:-1]

y=u[:,-1]





from sklearn import model_selection

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)

from sklearn.linear_model import LinearRegression

reg=LinearRegression()

reg.fit(X_train,y_train)

reg.coef_, reg.intercept_

res= reg.predict(X_test)

u = abs(y_test - res)

y=u/y_test

acc=1-y

"Linear Regression OLS R square",reg.score(X_test,y_test),"Accuracy",acc.mean()

from sklearn import linear_model

reg=linear_model.Ridge(alpha=.5)

reg.fit(X_train,y_train)

res= reg.predict(X_test)

u = abs(y_test - res)

y=u/y_test

acc=1-y

"Linear Regression Ridge R square",reg.score(X_test,y_test),"Accuracy" ,acc.mean()
reg = linear_model.Lasso(alpha=.1)

reg.fit(X_train,y_train)

res= reg.predict(X_test)

u = abs(y_test - res)

y=u/y_test

acc=1-y

"Linear Regression Lasso R square",reg.score(X_test,y_test),"Accuracy", acc.mean()
from sklearn.neighbors import KNeighborsRegressor

reg = KNeighborsRegressor()

reg.fit(X_train,y_train)

res= reg.predict(X_test)

u = abs(y_test - res)

y=u/y_test

acc=1-y

"KNN R square",reg.score(X_test,y_test),"Accuracy", acc.mean()
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(n_estimators=100)

reg.fit(X_train,y_train)

res= reg.predict(X_test)

u = abs(y_test - res)

y=u/y_test

acc=1-y

"Random Forest Regressor R square",reg.score(X_test,y_test),"Accuracy", acc.mean()
