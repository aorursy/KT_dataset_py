import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.linear_model import LinearRegression 

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import statsmodels.api as sm
startups=pd.read_csv("../input/50_Startups.csv")

df=startups.copy()
df.head()
df.info()
df.shape
df.isnull().sum()
corr=df.corr()

corr
sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
sns.scatterplot(df["R&D Spend"], df["Profit"]);
df.hist()
df.describe().T
df["State"].unique()
df_state=pd.get_dummies(df["State"])

df_state.head()
df = pd.concat([df, df_state], axis = 1)

df.head()
corr=df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
df.drop( ["State","California"], axis = 1, inplace = True)
df.head()
y=df["Profit"]
X=df.drop(["Profit"],axis=1)
y.head()
X.head()
X_train,X_test,y_train,y_test=train_test_split(X,y)
X_train.head()
X_test.head()
y_train.head()
y_test.head()
lm=LinearRegression()
model=lm.fit(X_train,y_train)
y_pred=model.predict(X_test)

y_pred
df_tahminler=pd.DataFrame(y_pred)

df_tahminler.columns=["y_pred"]

df_tahminler
df_gercekler=pd.DataFrame(y_test)

df_gercekler.columns=["y_gercek"]

df_gercekler
MAE=mean_absolute_error(y_test,y_pred)

MAE
MSE=mean_squared_error(y_test,y_pred)

MSE
RMSE=np.sqrt(mean_squared_error(y_test,y_pred))

RMSE
model.score(X_train,y_train)
lm=sm.OLS(y,X)
model=lm.fit()
model.summary()