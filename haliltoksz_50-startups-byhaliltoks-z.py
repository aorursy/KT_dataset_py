import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
startups = pd.read_csv('../input/50_Startups.csv')
startups.head()
startups.info()
startups.count()
startups.isna().sum()
startups.corr()
corr = startups.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values);
sns.scatterplot(x="R&D Spend", y="Profit", data=startups, color="magenta");
startups.hist(figsize = (15,15))
plt.show()
startups.describe()
startups["State"].unique()
startups_state = pd.get_dummies(startups["State"])
startups_state
df = pd.concat([startups, startups_state], axis = 1)
df
df.drop( ["State","California"], axis = 1, inplace = True)
df
y=df["Profit"]
X=df.drop(["Profit"],axis=1)
y
X
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y)
X_train
X_test
y_train
y_test
from sklearn.linear_model import LinearRegression 
lm=LinearRegression()
model=lm.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_pred
df_tahmin=pd.DataFrame(y_pred)
df_tahmin.columns=["y_pred"]
df_tahmin
from sklearn.metrics import mean_absolute_error
MAE=mean_absolute_error(y_test,y_pred)
MAE
from sklearn.metrics import mean_squared_error

MSE = mean_squared_error(y_test,y_pred)
MSE
import math

RMSE = math.sqrt(MSE)
RMSE
model.score(X_train,y_train)
import statsmodels.api as sm
lm=sm.OLS(y,X)
model=lm.fit()
model.summary()