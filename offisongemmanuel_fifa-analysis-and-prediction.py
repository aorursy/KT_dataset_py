import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
sns.set()
fifa=pd.read_csv("../input/fifa-dataset/data.csv")
fifa.head()
to_stay=["ID","Name","Age","Nationality","Club","Overall","Potential","Value","Wage","Real Face"]
fifa.drop(fifa.columns.difference(to_stay),axis="columns",inplace=True)
fifa.head()
fifa.set_index("ID",inplace=True)
fifa.head()
fifa.isnull().sum()
fifa["Club"].dropna(axis="rows",inplace=True)
fifa.dropna(axis="rows",inplace=True)
fifa.isnull().sum()
fifa.shape
fifa.head(20)
sns.boxplot(fifa["Age"])
q1=np.percentile(fifa["Age"],25)
q3=np.percentile(fifa["Age"],75)
iqr=q3-q1
lower=q1-(1.5*iqr)
upper=q3+(1.5*iqr)
fifa["Age"][(fifa["Age"]<np.abs(lower))|(fifa["Age"]>upper)].max()
fifa.Value.unique
fifa.info()
fifa["Value2"]=fifa["Value"].apply(lambda x: x.split("€")[1])

fifa["Value3"]=fifa["Value2"].apply(lambda x:x.split("M")[0]*1000000 if x.split("M")==True else x.split("K")[0]*1000)
fifa
fifa["Value3"]=fifa["Value2"].apply(lambda x: 1 if "M" in x else 0)
fifa["Value4"]=fifa["Value2"].apply(lambda x : x.split("M")[0] if "M" in x else x.split("K")[0]).astype(float)
fifa["Value5"]=(fifa[fifa["Value3"]==1]["Value4"]*1000000)
a=fifa[fifa["Value3"]==0]
fifa["Value5"].fillna(a["Value4"]*1000,inplace=True)
fifa
fifa.drop(["Value4","Value3","Value2","Value"],axis="columns",inplace=True)
fifa.rename(columns={"Value5":"Value"},inplace=True)

fifa[fifa["Value"]==60000]
fifa["Wage"].unique()
fifa["Wage2"]=fifa["Wage"].apply(lambda x: x.split("€")[1])
fifa["Wage3"]=fifa["Wage2"].apply(lambda x:x.split("K")[0]).astype(float)*1000
fifa.head()
fifa.drop(["Wage","Wage2"],axis="columns",inplace=True)
fifa.rename(columns={"Wage3":"Wage"},inplace=True)
fifa
fifa["Nationality"].value_counts()
fifa["Nationality"].unique()
fifa["RF_Dummy"]=fifa["Real Face"].apply(lambda x: 1 if x=="Yes" else 0)
fifa
rf_used=np.sum(fifa["RF_Dummy"]==1)
rf_notused=np.sum(fifa["RF_Dummy"]==0)
(rf_used/fifa.shape[0])*100
(rf_notused/fifa.shape[0])*100
plt.hist(fifa["Real Face"])
sns.barplot(fifa["Real Face"],fifa["Overall"])
fifa
%matplotlib
sns.barplot(fifa["Overall"],fifa["RF_Dummy"])
%matplotlib inline
sns.regplot(fifa["Overall"],fifa["Wage"])
from scipy.stats import pearsonr
pearsonr(fifa["Overall"],fifa["Wage"])
fifa
sns.regplot(fifa["Overall"],fifa["Value"])
pearsonr(fifa["Overall"],fifa["Value"])
fifa.corr()
fifa[(fifa["Overall"]>60)&(fifa["Overall"]<80)]
africa=["Senegal" , "Egypyt","Gabon","Morocco","Algeria","Guinea","Ghana","Central African Rep.", "DR Congo","Ivory Coast",  "Mali","Nigeria", "Cameroon", "Kenya","Cape Verde", "Togo","Zimbabwe","Angola","Burkina Faso", "Tunisia","Equatorial Guinea", "Guinea Bissau","South Africa","Madagascar","Tanzania","Gambia","Benin","Congo","Mozambique","Sierra Leone","Zambia","Chad","Libya","Eritrea","Uganda","Niger","Mauritania","Namibia","Sudan","Ethiopia","Rwanda","South Sudan"]
fifa_africa=fifa[fifa["Nationality"].isin(africa)]
fifa_africa.corr()
sns.scatterplot(fifa_africa["Overall"],fifa_africa["Value"])
sns.regplot(fifa_africa["Overall"],fifa_africa["Value"])
sns.regplot(fifa_africa["Wage"],fifa_africa["Value"])
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
model=LinearRegression()
fifa.head()
fifa=fifa.reset_index()
X=fifa[["Age","Overall","Potential","Wage"]]
y=fifa["Value"]
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=0)
models={"lr":{
    "model":LinearRegression()
},
       "svr":{
           "model":SVR()
       },
       "dtr":{
           "model":DecisionTreeRegressor()
       },
       "gbr":{
           "model":GradientBoostingRegressor()
       },
       "rtr":{
           "model":RandomForestRegressor()
       }}
for name,model in models.items():
    pre=model["model"].fit(Xtrain,ytrain)
    print(f"The accuracy of {name} is {pre.score(Xtest,ytest)}")
gbr=GradientBoostingRegressor().fit(Xtrain,ytrain)
print(f"Your training accuracy is {gbr.score(Xtrain,ytrain)} while your test accuracy is {gbr.score(Xtest,ytest)}")
ypred=gbr.predict(Xtest)
gbr.predict([[20,76,78,50000]])
mean_absolute_error(ytest,ypred)
mean_squared_error(ytest,ypred)
np.sqrt(mean_squared_error(ytest,ypred))