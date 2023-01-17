import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt #Veriyi görselleştirme kütüphanesi(genellikle 2 boyutta)
startups = pd.read_csv("../input/startup-ds/50_Startups.csv")
df = startups.copy()
df.head() #Veri çerçevesindeki ilk 5 veriyi gözlemlemek için head() çağırılır.
df.info()
df.shape
df.isnull().sum()
df.corr()
corr=df.corr()
sns.heatmap(corr,xticklabels=corr.columns.values,
                 yticklabels=corr.columns.values);
sns.scatterplot(x="R&D Spend",y="Profit",data=df,color="black");
df.hist();
df.describe()
df["State"].unique()
enc_State = pd.get_dummies(df,columns=["State"],prefix = df.State[0:1]);
enc_State
encState = enc_State.drop(['New York_New York','New York_California'],axis=1)
encState.columns = ['R&D Spend', 'Administration', 'Marketing Spend', 'Profit',
       'CityInfo']
df = encState
bagimsiz_deg = ["R&D Spend","Marketing Spend"]
df_bag=df["Profit"]
df_bagimsiz = df[bagimsiz_deg]
df_bag
df_bagimsiz
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df_bagimsiz,df_bag,test_size=0.25,random_state=123)
X_train
y_train
X_test
y_test
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
model=linear_reg.fit(df_bagimsiz,df_bag)
X_test
y_pred = model.predict(X_test)[0:13]
y_pred
df_result= pd.DataFrame({"y_pred" : y_pred , "y_test":y_test , "Hata miktari" : y_test - y_pred})
df_result
from sklearn.metrics import mean_absolute_error , mean_squared_error
MSE = mean_squared_error(y_test,y_pred)
MAE = mean_absolute_error(y_test,y_pred)
RMSE = mean_squared_error(y_test,y_pred,squared=False)
print("MSE :" , "%.4f" % MSE,
     "\nMAE :" , "%.4f" % MAE,
     "\nRMSE :","%.4f" % RMSE)
model.score(X_train,y_train)
import statsmodels.api as sm
y = df.iloc[:,-2]
X = df[["R&D Spend","Marketing Spend"]]
lm = sm.OLS(y,X)
model = lm.fit()
model.conf_int()