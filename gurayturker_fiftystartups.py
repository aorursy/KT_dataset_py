import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import missingno as msno

from sklearn.linear_model import LinearRegression 

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import statsmodels.api as sm
startups=pd.read_csv("../input/50_Startups.csv",sep=",")
df=startups.copy()
df.head()
df.info()
df.describe().T
df.shape
df.ndim
df.size
df.count()
df["State"].unique()
df.isnull().sum()
msno.bar(df)
corr=df.corr()

corr
sns.set(font_scale=1.15)

plt.figure(figsize=(14, 10))



sns.heatmap(corr, vmax=.8, linewidths=0.01,

            square=True,annot=True,cmap='YlGnBu',linecolor="black")

plt.title('Değişkenlerin Birbiri ile Korelasyonu');
sns.scatterplot(x="R&D Spend",y="Profit",data=df)
sns.jointplot(x=df["R&D Spend"],y=df["Profit"],data=df,joint_kws={"s":5,"color":"red"},marginal_kws={"color":"red"})
sns.lmplot(x="R&D Spend",y="Profit",data=df,col="State",scatter_kws={"color":"green"},

          line_kws={"color":"red"})
df["R&D Spend"].hist()
df.hist()
df=pd.get_dummies(df,columns=["State"],prefix=["State"])
df.drop(["State_California"],axis=1,inplace=True)
X=df.drop(["Profit"],axis=1)

X.head()
y=df[["Profit"]]

y.head()
sns.jointplot(x="R&D Spend",y="Profit",data=df,kind="reg")
sns.jointplot(x="Administration",y="Profit",data=df,kind="reg")
sns.jointplot(x="Marketing Spend",y="Profit",data=df,kind="reg")
reg=LinearRegression()
model=reg.fit(X,y)

model
model.intercept_
model.coef_
model.score(X,y)
g=sns.regplot(df["R&D Spend"],df["Profit"],ci=None,scatter_kws={"color":"r","s":9})

g.set_xlabel("R&D Spend")

g.set_ylabel("Profit")
g=sns.regplot(df["Marketing Spend"],df["Profit"],ci=None,scatter_kws={"color":"r","s":9})

g.set_xlabel("Marketing Spend")

g.set_ylabel("Profit")
g=sns.regplot(df["Administration"],df["Profit"],ci=None,scatter_kws={"color":"r","s":9})

g.set_xlabel("R&D Spend")

g.set_ylabel("Profit")
model.predict(X)
gercek_y=y
tahmin_edilen_y=pd.DataFrame(model.predict(X))
hatalar=pd.concat([gercek_y,tahmin_edilen_y],axis=1)

hatalar.columns=["gercek_y","tahmin_edilen_y"]

hatalar
hatalar["hata"]=hatalar["tahmin_edilen_y"]-hatalar["gercek_y"]
hatalar
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=99)
X_train.head()
X_test.head()
y_train.head()
y_test.head()
lm=LinearRegression()
model=lm.fit(X_train,y_train)
model.predict(X_train)
model.coef_
model.intercept_
y_pred=model.predict(X_test)
y_pred_df=pd.DataFrame(model.predict(X_test))
y_pred_df.columns=["y_pred"]

y_pred_df
df_yeni=pd.DataFrame(y_pred)

df_yeni["y_test"]=y_test.values

df_yeni.columns=["Tahmin","Gerçek"]

df_yeni
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