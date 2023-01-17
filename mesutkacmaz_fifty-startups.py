import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt
startups = pd.read_csv("../input/fiftystartups/fiftystartups.csv")

df = startups.copy()
df.head(5)
df.info()
df.shape
df.isna().sum()
df.corr()
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
sns.scatterplot(x = "R&D Spend", y = "Profit", data = df, color="blue");
df.hist(bins=12)
df.describe().T
df.State.unique()
df_State = pd.get_dummies(df["State"])
df_State.head()
df_State.columns = ['California','Florida','New York']

df_State.head()
df = pd.concat([df, df_State], axis = 1)

df.drop(["California", "State"], axis = 1, inplace = True)

df.head()
Y = df.iloc[:,3].values

X = df[['R&D Spend', 'Administration', 'Marketing Spend']]
X
Y
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 42, shuffle=1)
X_train
X_test
Y_train
Y_test
from sklearn.linear_model import LinearRegression

model=LinearRegression()
model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)
df1 = pd.DataFrame({"Gerçek" : Y_test, "Tahmin Edilen" : Y_pred})



df1
import sklearn.metrics as metrics



mae = metrics.mean_absolute_error(Y_test, Y_pred)

mse = metrics.mean_squared_error(Y_test, Y_pred)

rmse = np.sqrt(mse)



print("Mean Absolute Rrror(MAE):",mae)

print("Mean Squared Error (MSE):", mse)

print("Root Mean Squared Error (RMSE):", rmse)
model.score(X_train, Y_train)
import statsmodels.api as sm
model = sm.OLS(Y, X).fit()
print(model.summary())