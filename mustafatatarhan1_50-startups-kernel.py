import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import statsmodels.api as sm
startups=pd.read_csv("../input/50-startups/50_Startups.csv")
df=startups.copy()
df.head()
df.info()
df.shape
df.isnull().sum()
corr=df.corr()
corr
corr=df.corr()
sns.heatmap(corr,annot=True,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values);
sns.scatterplot(x="R&D Spend",y="Profit",data=df);
df.hist()
df.describe().T
df["State"].unique()
df_state=pd.get_dummies(df["State"])
df_state.head()
df = pd.concat([df, df_state], axis = 1)
df.head()
df.drop( ["State","New York"], axis = 1, inplace = True)
df.head()
X = df.drop("Profit", axis = 1)
Y = df["Profit"]
X.head()
Y.head()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 3512, shuffle=1)
X_train
X_test
Y_train
Y_test
lm=LinearRegression()
model = lm.fit(X_train, Y_train)
Y_pred = lm.predict(X_test)
df_comp = pd.DataFrame({'Olağan': Y_test, 'Tahmini': Y_pred})
df_comp
MAE=mean_absolute_error(Y_test,Y_pred)
MAE
MSE=mean_squared_error(Y_test,Y_pred)
MSE
RMSE=np.sqrt(mean_squared_error(Y_test,Y_pred))
RMSE
model.score(X, Y)
stmodel = sm.OLS(Y, X).fit()
stmodel.summary()


