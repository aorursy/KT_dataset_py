import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
df = pd.read_csv("/kaggle/input/50-startups/50_Startups.csv")

df.head()
df.info()

df.shape
df.isna().sum()
df.corr()
corr = df.corr()
sns.heatmap(corr,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values);
sns.scatterplot(x= "R&D Spend", y="Profit", data=df);
df.hist()
df.describe()
df["State"].unique()
pd.get_dummies(df, ('State'))



pd.get_dummies(df,drop_first = True)

X = df.drop("Profit", axis = 1)
y = df["Profit"]
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=99)
X_train.head()
X_test.head()
y_train.head()
y_test.head()
lm = LinearRegression()
model = lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
df_a = pd.DataFrame({'Gercek': y_test, 'Tahmin': y_pred})
df_a
MAE = mean_absolute_error(y_test, y_pred)
MAE
MSE = mean_squared_error(y_test, y_pred)
MSE

RMSE = math.sqrt(MSE)
RMSE

model.score(X_train, y_train)
model.score(X_test, y_test)




