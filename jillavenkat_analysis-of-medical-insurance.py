#Loading libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os 
print(os.getcwd())

data=pd.read_csv("../input/insurance.csv")


print(data.columns)
print("\n")
print("*"*40)
print("\n")
print(data.head())
data.info()
f,ax=plt.subplots(1,2,figsize=(18,5))

g=sns.countplot(y=data.smoker,order=data.smoker.value_counts().index,ax=ax[0])
for i,v in enumerate(data.smoker.value_counts().values):
    g.text(0.5,i,v,weight="bold")
ax[0].set_title("Smoker")
ax[0].set_ylabel("")
    
f=sns.countplot(y=data.sex,order=data["sex"].value_counts().index,ax=ax[1])
for i,v in enumerate(data.sex.value_counts().values):
    f.text(0.5,i,v,weight="bold")
ax[1].set_title("Gender")
ax[1].set_ylabel("")
plt.subplots_adjust(wspace=0.8)


print(data.columns)

f,ax=plt.subplots(1,2,figsize=(15,8))

g1=data.groupby(["sex","smoker"])["age"].count().to_frame().reset_index()
g1.pivot("sex","smoker","age").plot(kind="barh",ax=ax[0])
ax[0].set_title("Smoker vs Gender",weight="bold",size=20)
ax[0].set_ylabel("")

g2=data.groupby(["sex","region"])["age"].count().to_frame().reset_index()
g2.pivot("region","sex","age").plot(kind="barh",ax=ax[1])
ax[1].set_title("Regionwise gender",weight="bold",size=20)
ax[1].set_ylabel("")


plt.legend(loc="lower right")


g=sns.countplot(y=data.region,order=data["region"].value_counts().index)
for i,v in enumerate(data.region.value_counts().values):
    g.text(0.5,i,v,weight="bold")
    
plt.title("Regions",size=20,weight="bold")
plt.ylabel("")
plt.xlabel("")
fig=plt.gcf()
fig.set_size_inches(15,8)

f,ax=plt.subplots(1,2,figsize=(18,8))

data["sex"].value_counts().plot.pie(autopct="%1.1f%%",subplots=True,ax=ax[0])
ax[0].set_ylabel("")
ax[0].set_title("Gender Ratio",size=15,weight="bold")
fig=plt.gcf()
my_circle=plt.Circle((0,0),0.7,color="white")
fig.gca().add_artist(my_circle)


data["region"].value_counts().plot.pie(autopct="%1.1f%%",ax=ax[1],subplots=True,explode=[0,0.05,0.01,0])

ax[1].set_ylabel("")
ax[1].set_title("Region",size=15,weight="bold")


print(data.columns)

sns.lmplot(x="bmi",y="charges",hue="smoker",data=data,fit_reg=False,palette="magma")
fig=plt.gcf()
fig.set_size_inches(15,8)
plt.title("Plot for Charges & BMI",weight="bold")

f,ax=plt.subplots(1,2,figsize=(15,8))
sns.barplot(y="smoker",x="charges",hue="sex",data=data,ax=ax[0])

ax[0].set_title("Charges for Smoker",weight="bold",size=15)
ax[0].set_ylabel("")
ax[0].set_xlabel("")

sns.barplot(y="sex",x="charges",data=data,ax=ax[1])
ax[1].set_title("Charges for Gender",weight="bold",size=15)
ax[1].set_xlabel("")
ax[1].set_ylabel("")
plt.subplots_adjust(wspace=1.0)
f,ax=plt.subplots(1,2,figsize=(15,5))
sns.boxplot(y="smoker",x="charges",data=data[data["smoker"]=="yes"],hue="sex",palette="magma",ax=ax[0])
ax[0].set_title("Chargers for Smokers ",size=20,weight="bold")

sns.boxplot(y="smoker",x="charges",data=data[data["smoker"]=="no"],hue="sex",palette="magma",ax=ax[1])
ax[1].set_title("Chargers for non-Smokers ",weight="bold",size=20)
sns.lmplot(y="charges",x="age",hue="smoker",data=data,fit_reg=False)
fig=plt.gcf()
fig.set_size_inches(10,8)
plt.title("Plot for Charges and Age",size=20,weight="bold")
plt.xlabel("Age",size=10,weight="bold")
plt.ylabel("Charges",size=10,weight="bold")
cat=data.select_dtypes(include="object").columns
num=data.select_dtypes(exclude="object").columns
#Finding Correlation Matrix

sns.heatmap(data.corr(),annot=True,cbar=True)
plt.subplots(figsize=(10,8))
sns.boxplot(x="charges",y="smoker",data=data[data["age"]<=18])
plt.title("Charges for smokers leass than 18 Yrs of Age")



data["sex"]=data["sex"].map({"female":0,"male":1})
data["smoker"]=data["smoker"].map({"yes":1,"no":0})

r1=pd.get_dummies(data["region"])
fin=pd.merge(data,r1,how="left",left_index=True,right_index=True)
fin.drop("region",axis=1,inplace=True)


from sklearn.preprocessing import StandardScaler,MinMaxScaler
st=StandardScaler()
scaler=MinMaxScaler(feature_range=(0,1))
a=np.array(data['bmi'])
print(a)
a1=a.reshape(-1,1)
scled=scaler.fit_transform(a1)
fin1=fin.copy()
fin1["bmi_scale"]=pd.DataFrame(scled)

fin1.drop("bmi",axis=1,inplace=True)
fin1.head()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


#Splitting the data
X=fin.drop("charges",axis=1)
y=fin["charges"]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
#LINear regression
Train_rmse=[]
Test_rmse=[]

from sklearn.linear_model import LinearRegression
mod_lg=LinearRegression()
mod_lg.fit(X_train,y_train)
pre_lg=mod_lg.predict(X_train)
pred_lg=mod_lg.predict(X_test)


print("on Trained Data")
print("Mean_Squared Error:",mean_squared_error(pre_lg,y_train))
print("Absolute Error:",mean_absolute_error(pre_lg,y_train))
print("R2_score:",r2_score(pre_lg,y_train))


print("*"*40)

print("On Test Data")
print("Mean_Squared Error:",mean_squared_error(pred_lg,y_test))
print("Absolute Error:",mean_absolute_error(pred_lg,y_test))
print("R2_score:",r2_score(pred_lg,y_test))

from sklearn.ensemble import RandomForestRegressor
mod_rf=RandomForestRegressor(n_estimators=500)
print(mod_rf)
mod_rf.fit(X_train,y_train)
pre_rf=mod_rf.predict(X_train)
pred_rf=mod_rf.predict(X_test)


print("on Trained Data")
print("Mean_Squared Error:",mean_squared_error(pre_rf,y_train))
print("Absolute Error:",mean_absolute_error(pre_rf,y_train))


print("*"*40)

print("On Test Data")
print("Mean_Squared Error:",mean_squared_error(pred_rf,y_test))
print("Absolute Error:",mean_absolute_error(pred_rf,y_test))

#Plotting
plt.figure(figsize=(10,6))

plt.scatter(pre_rf,pre_rf - y_train,
          c = 'black', marker = 'o', s = 35, alpha = 0.5,
          label = 'Train data')
plt.scatter(pred_rf,pred_rf - y_test,
          c = 'c', marker = 'o', s = 35, alpha = 0.7,
          label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Tailings')
plt.legend(loc = 'upper left')
plt.hlines(y = 0, xmin = 0, xmax = 60000, lw = 2, color = 'red')
plt.show()


feature_importances=pd.DataFrame(mod_rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False).reset_index()
feature_importances.columns=["feat","values"]
sns.barplot(y=feature_importances.feat,x=feature_importances["values"])
from sklearn.ensemble import AdaBoostRegressor
mod_ada=AdaBoostRegressor()


mod_ada.fit(X_train,y_train)
pre_ada=mod_ada.predict(X_train)
pred_ada=mod_ada.predict(X_test)


print("on Trained Data")
print("Mean_Squared Error:",mean_squared_error(pre_ada,y_train))
print("Absolute Error:",mean_absolute_error(pre_ada,y_train))


print("*"*40)

print("On Test Data")
print("Mean_Squared Error:",mean_squared_error(pred_ada,y_test))
print("Absolute Error:",mean_absolute_error(pred_ada,y_test))


feature_importances=pd.DataFrame(mod_ada.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False).reset_index()
feature_importances.columns=["feat","values"]
sns.barplot(y=feature_importances.feat,x=feature_importances["values"])
from sklearn.ensemble import GradientBoostingRegressor
mod_gbm=GradientBoostingRegressor()


mod_gbm.fit(X_train,y_train)
pre_gbm=mod_gbm.predict(X_train)
pred_gbm=mod_gbm.predict(X_test)


print("on Trained Data")
print("Mean_Squared Error:",mean_squared_error(pre_gbm,y_train))
print("Absolute Error:",mean_absolute_error(pre_gbm,y_train))


print("*"*40)

print("On Test Data")
print("Mean_Squared Error:",mean_squared_error(pred_gbm,y_test))
print("Absolute Error:",mean_absolute_error(pred_gbm,y_test))

feature_importances=pd.DataFrame(mod_gbm.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False).reset_index()
feature_importances.columns=["feat","values"]
sns.barplot(y=feature_importances.feat,x=feature_importances["values"])
